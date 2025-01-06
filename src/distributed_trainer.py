# Standard library imports
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Third-party imports
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch._prims_common import DeviceLikeType
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Local imports
from src.data_utils import MyTrainDataset
from src.dist_strategy.ddp_strategy import DDPStrategy
from src.dist_strategy.dist_strategy import DistributedStrategy
from src.dist_strategy.fsdp_strategy import FSDPStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    max_epochs: int
    save_every: int
    batch_size: int
    learning_rate: float
    snapshot_path: Path
    device: str = "auto"
    parallel_strategy: str = "ddp"


class DistributedEnvironment:
    """Manages distributed training setup and environment variables."""

    def __init__(self, device: str = "auto"):
        self.device = self._determine_device(device)
        self.is_gpu = self.device == "cuda"
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = (
            int(os.environ.get("LOCAL_RANK", 0)) if self.is_gpu else 0
        )
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    @staticmethod
    def _determine_device(device: str) -> DeviceLikeType:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def setup(self) -> None:
        backend = "nccl" if self.device == "cuda" else "gloo"
        init_process_group(backend=backend)

        if self.device == "cuda":
            if "LOCAL_RANK" not in os.environ or "RANK" not in os.environ:
                raise ValueError(
                    "LOCAL_RANK and RANK environment variables must be set "
                    "for GPU training"
                )
            torch.cuda.set_device(self.local_rank)


class ModelCheckpoint:
    """Handles model checkpointing operations."""

    def __init__(
        self,
        path: Path,
        env: DistributedEnvironment,
        strategy: DistributedStrategy,
    ):
        self.path = path
        self.env = env
        self.strategy = strategy

    def save(self, model: torch.nn.Module, epoch: int) -> None:
        if self.env.global_rank == 0:
            state_dict = {
                "MODEL_STATE": model.state_dict(),
                "EPOCHS_RUN": epoch,
            }
            self.strategy.save_checkpoint(model, state_dict, str(self.path))
            logger.info(
                f"Epoch {epoch} | Training snapshot saved at {self.path}"
            )

    def load(
        self, model: torch.nn.Module, device: DeviceLikeType
    ) -> Tuple[torch.nn.Module, int]:
        if not self.path.exists():
            return model, 0

        logger.info("Loading snapshot")
        snapshot = torch.load(self.path, map_location=torch.device(device))
        return self.strategy.load_checkpoint(model, snapshot, device)


class Trainer:
    """Manages the training process in a distributed environment."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        env: DistributedEnvironment,
    ) -> None:
        self.env = env
        self.config = config

        self.model = self._setup_model(model)

        self.checkpoint = ModelCheckpoint(
            Path(config.snapshot_path), env, self._strategy
        )
        self.train_data = train_data
        self.optimizer = optimizer
        self.epochs_run = 0

        logger.info(
            f"[{str(self.env.device).upper()}{self.env.global_rank}] "
            f"Running with {self.env.world_size} processes total"
        )

    def _setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        device = self.env.local_rank if self.env.is_gpu else self.env.device
        model = model.to(device)

        # To give type hint to the MyPy
        self._strategy: DistributedStrategy

        if self.config.parallel_strategy.lower() == "fsdp":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "FSDP requires CUDA to be available. "
                    "Please use DDP for CPU training."
                )
            self._strategy = FSDPStrategy(self.env.is_gpu)
        else:
            self._strategy = DDPStrategy(self.env.local_rank, self.env.is_gpu)

        self.checkpoint = ModelCheckpoint(
            Path(self.config.snapshot_path), self.env, self._strategy
        )
        model = self._strategy.prepare_model(model, device)
        model, self.epochs_run = self.checkpoint.load(model, device)
        return model

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int) -> None:
        b_sz = len(next(iter(self.train_data))[0])
        logger.info(
            f"[{str(self.env.device).upper()}{self.env.global_rank}] "
            f"Epoch {epoch} | Batchsize: {b_sz} | "
            f"Steps: {len(self.train_data)}"
        )
        assert isinstance(self.train_data.sampler, DistributedSampler)
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(
                self.env.local_rank if self.env.is_gpu else self.env.device
            )
            targets = targets.to(
                self.env.local_rank if self.env.is_gpu else self.env.device
            )
            self._run_batch(source, targets)

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if (
                self.env.local_rank == 0
                and epoch % self.config.save_every == 0
            ):
                self.checkpoint.save(self.model, epoch)


def load_train_objs(
    cfg: DictConfig,
) -> Tuple[Dataset, torch.nn.Module, torch.optim.Optimizer]:
    train_set = MyTrainDataset(cfg.train.dataset_size)
    model = torch.nn.Linear(cfg.model.input_size, cfg.model.output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.learning_rate)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging to write to both file and console."""
    logger.info(f"Logging setup complete. Log file: " f"{cfg.logging.file}")
    log_file = Path(cfg.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create formatters and handlers
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(cfg.logging.level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging setup complete. Log file: {log_file}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    try:
        # Setup logging first
        setup_logging(cfg)

        train_config = TrainingConfig(
            max_epochs=cfg.train.total_epochs,
            save_every=cfg.train.save_every,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            snapshot_path=cfg.train.snapshot_path,
            device=cfg.train.get("device", "auto"),
            parallel_strategy=cfg.train.get("parallel_strategy", "ddp"),
        )

        # Set up distributed environment
        env = DistributedEnvironment(train_config.device)
        env.setup()

        dataset, model, optimizer = load_train_objs(cfg)
        train_data = prepare_dataloader(dataset, train_config.batch_size)

        trainer = Trainer(model, train_data, optimizer, train_config, env)
        trainer.train(train_config.max_epochs)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Only call destroy_process_group if it was initialized
        if torch.distributed.is_initialized():
            destroy_process_group()


if __name__ == "__main__":
    main()
