from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch._prims_common import DeviceLikeType
from torch.nn import Module


class DistributedStrategy(ABC):
    @abstractmethod
    def prepare_model(self, model: Module, device: Any) -> Module:
        """Wrap the model with the appropriate parallel strategy."""
        pass

    @abstractmethod
    def save_checkpoint(
        self, model: Module, state_dict: dict, path: str
    ) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(
        self, model: Module, checkpoint: dict, device: DeviceLikeType
    ) -> Tuple[Module, int]:
        """Load model checkpoint."""
        pass
