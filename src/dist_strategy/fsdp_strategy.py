from typing import Any, Tuple

import torch
from torch._prims_common import DeviceLikeType
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from .dist_strategy import DistributedStrategy


class FSDPStrategy(DistributedStrategy):
    def __init__(self, is_gpu: bool):
        self.is_gpu = is_gpu

    def prepare_model(
        self, model: torch.nn.Module, device: Any
    ) -> torch.nn.Module:
        return FSDP(
            model,
            device_id=torch.cuda.current_device() if self.is_gpu else None,
            cpu_offload=(
                CPUOffload(offload_params=True) if not self.is_gpu else None
            ),
        )

    def save_checkpoint(
        self, model: torch.nn.Module, state_dict: dict, path: str
    ) -> None:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            state_dict["MODEL_STATE"] = model.state_dict()
            torch.save(state_dict, path)

    def load_checkpoint(
        self, model: torch.nn.Module, checkpoint: dict, device: DeviceLikeType
    ) -> Tuple[torch.nn.Module, int]:
        load_policy = FullStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, load_policy
        ):
            model.load_state_dict(checkpoint["MODEL_STATE"])
        return model, checkpoint["EPOCHS_RUN"]
