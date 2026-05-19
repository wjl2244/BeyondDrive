# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union
import hydra
import time
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score, pdm_score_full_v1
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = args[0]['model_trajectory']

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
            simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    # agent: AbstractAgent = instantiate(cfg.agent)
    # agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=None,
    )

    save_dir = Path(cfg.output_dir) / "negative_sample_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[pd.DataFrame] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            pred_trajectorys = model_trajectory[token]['pred_trajectorys_array']  # pred_trajectorys
            pdm_result = pdm_score_full_v1(
                metric_cache=metric_cache,
                vocab_trajectories=pred_trajectorys,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update({
                "pred_trajectorys": pred_trajectorys,
                "pdm_score_matrix": pdm_result,
            })
            with open(f'{save_dir}/{token}.pkl', 'wb') as f:
                pickle.dump(score_row, f)
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        # pdm_results.append(score_row)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    build_logger(cfg)

    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    model_outputs_save_path = os.path.join(cfg.output_dir, cfg.agent.checkpoint_path.split('/')[-1].split('.ckpt')[-2] + '.pkl')
    
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    dataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=scene_filter.log_names,
        )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks(), logger=False)
    predictions = trainer.predict(
        AgentLightningModule(agent=agent),
        dataloader,
        return_predictions=True
    )

    dist.barrier()
    all_predictions = [None for _ in range(dist.get_world_size())]

    if dist.is_initialized():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions.append(predictions)

    if dist.get_rank() != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)
    
    dump_path = os.path.join(model_outputs_save_path)
    pickle.dump(merged_predictions, open(dump_path, 'wb'))
    print(f'Subscore/Trajectories saved to {dump_path}')
    
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    print(f"Processing {len(data_points)} scenarios...")

    worker = build_worker(cfg)

    start_time = time.time()
    
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)
    
    print("==================================")
    print(f"Time taken: {time.time() - start_time}")
    print("==================================")


if __name__ == "__main__":
    main()
