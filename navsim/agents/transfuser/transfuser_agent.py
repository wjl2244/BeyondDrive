from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.utils import WarmupCosLR
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        max_epochs: int = 100,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self.max_epochs = max_epochs

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(config)
        
        from navsim.agents.pdm_score import ComputePDMScore
        self.pdm_score_computer = ComputePDMScore(cache_path='navtest_v1_metric_cache')

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # return SensorConfig.build_all_sensors(include=[3])
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=[3],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""

        if self.training:
            loss_dict = transfuser_loss(targets, predictions, self._config)
        else:
            loss_dict = {"loss": 0}
            if self.pdm_score_computer is not None:
                pdm_scores = self.pdm_score_computer.get_pdm_scores(predictions["trajectory"], targets["token"])
                loss_dict.update(pdm_scores)
        
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        if self._config.scheduler_type == 'constant':
            return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
        elif self._config.scheduler_type == 'warmup_cos':
            backbone_params_name = '_backbone.image_encoder'
            img_backbone_params = list(
                filter(lambda kv: backbone_params_name in kv[0], self.named_parameters()))
            default_params = list(
                filter(lambda kv: backbone_params_name not in kv[0], self.named_parameters()))
            params_lr_dict = [
                {'params': [tmp[1] for tmp in default_params]},
                {
                    'params': [tmp[1] for tmp in img_backbone_params],
                    'lr': self._lr * 1,
                    'weight_decay': 0
                }
            ]
            optim = torch.optim.AdamW(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
            scheduler = WarmupCosLR(
                optim,
                min_lr=1e-7,
                epochs=self.max_epochs,
                warmup_epochs=3,
            )
            return{
                "optimizer": optim,
                "lr_scheduler": scheduler
            }

    def get_training_callbacks(self) -> List[pl.Callback]:

        val_score = "val_score"
        filename = "{epoch:02d}-{val_score:.3f}"
        mode='max'

        ckpt_callback = ModelCheckpoint(
            save_top_k=5,
            monitor=val_score,
            mode=mode,
            dirpath=f"{self._config.output_dir}/",
            filename=filename,
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        # return [ckpt_callback, lr_monitor, TransfuserCallback(self._config)]
        return [ckpt_callback, lr_monitor]
