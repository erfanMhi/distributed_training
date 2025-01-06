from typing import Any, Tuple

import torch
from torch._prims_common import DeviceLikeType
from torch.nn.parallel import DistributedDataParallel as DDP

from .dist_strategy import DistributedStrategy


class DDPStrategy(DistributedStrategy):
    def __init__(self, local_rank: int, is_gpu: bool):
        self.local_rank = local_rank
        self.is_gpu = is_gpu

    def prepare_model(
        self, model: torch.nn.Module, device: Any
    ) -> torch.nn.Module:
        return DDP(
            model,
            device_ids=[self.local_rank] if self.is_gpu else None,
        )

    def save_checkpoint(
        self, model: torch.nn.Module, state_dict: dict, path: str
    ) -> None:
        torch.save(state_dict, path)

    def load_checkpoint(
        self, model: torch.nn.Module, checkpoint: dict, device: DeviceLikeType
    ) -> Tuple[torch.nn.Module, int]:
        model.load_state_dict(checkpoint["MODEL_STATE"])
        return model, checkpoint["EPOCHS_RUN"]
