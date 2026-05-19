from typing import Tuple
from pathlib import Path
import logging
import random
import numpy as np
import hydra

import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datetime
# import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
# import swanlab as wandb
# from swanlab.integration.pytorch_lightning import SwanLabLogger as WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

seed = 0
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
np.random.seed(seed) # Numpy module.
random.seed(seed) # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    test_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    test_scene_filter.log_names = cfg.test_logs
    
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    test_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=test_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    test_data = Dataset(
        scene_loader=test_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, test_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    cfg.agent.max_epochs = cfg.trainer.params.max_epochs
    cfg.agent.config.output_dir = cfg.output_dir
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            is_training=True,
            use_beyonddrive=agent._config.use_beyonddrive,
            negative_samples_path=cfg.negative_samples_path,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.test_logs,
            is_training=False,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
    print("len(val_data):",len(val_data))
    print("len(train_data):",len(train_data))

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    # wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # wandb_logger = WandbLogger(
    #     name=cfg.experiment_uid,
    #     project=cfg.experiment_name,
    #     config=wandb_config,
    #     settings=wandb.Settings(init_timeout=1200)
    # )

    if 'debug' in cfg.experiment_name:
        wandb_logger = None
    else:
        wandb_logger = TensorBoardLogger(
            save_dir=f"{cfg.output_dir}/tensorboard_logs/",
            name=cfg.experiment_uid,
            default_hp_metric=False,
        )

    logger.info("Building Trainer")
    
    if cfg.trainer.params.get('strategy') is not None:
        cfg.trainer.params.__delattr__('strategy')
    trainer = pl.Trainer(**cfg.trainer.params,
                            callbacks=agent.get_training_callbacks(),
                            logger=wandb_logger,
                            strategy=DDPStrategy(
                                static_graph=False, 
                                timeout=datetime.timedelta(seconds=3600),
                                find_unused_parameters=True,)
                    )
    
    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
