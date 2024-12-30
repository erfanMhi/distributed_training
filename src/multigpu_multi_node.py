import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data_utils import MyTrainDataset

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

class DistributedEnvironment:
    """Manages distributed training setup and environment variables."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._determine_device(device)
        self.is_gpu = self.device == "cuda"
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_gpu else 0
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    @staticmethod
    def _determine_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    def setup(self) -> None:
        
        backend = "nccl" if self.device == "cuda" else "gloo"
        init_process_group(backend=backend)
        
        if self.device == "cuda":
            if "LOCAL_RANK" not in os.environ or "RANK" not in os.environ:
                raise ValueError(
                    "LOCAL_RANK and RANK environment variables must be set for GPU training"
                )
            torch.cuda.set_device(self.local_rank)
            

class ModelCheckpoint:
    """Handles model checkpointing operations."""
    
    def __init__(self, path: Path):
        self.path = path

    def save(self, model: torch.nn.Module, epoch: int) -> None:
        snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.path)
        logger.info(f"Epoch {epoch} | Training snapshot saved at {self.path}")

    def load(self, model: torch.nn.Module, device: str) -> Tuple[torch.nn.Module, int]:
        if not self.path.exists():
            return model, 0
            
        logger.info("Loading snapshot")
        snapshot = torch.load(self.path, map_location=device)
        model.load_state_dict(snapshot["MODEL_STATE"])
        return model, snapshot["EPOCHS_RUN"]

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
        self.checkpoint = ModelCheckpoint(Path(config.snapshot_path))
        
        self.model = self._setup_model(model)
        self.train_data = train_data
        self.optimizer = optimizer
        self.config = config
        self.epochs_run = 0

        logger.info(f"[{self.env.device.upper()}{self.env.global_rank}] "
                   f"Running with {self.env.world_size} processes total")

    def _setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        device = self.env.local_rank if self.env.is_gpu else self.env.device
        model = model.to(device)
        model, self.epochs_run = self.checkpoint.load(model, device)
        return DDP(model, device_ids=[self.env.local_rank] if self.env.is_gpu else None)

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        logger.info(f"[{self.env.device.upper()}{self.env.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.env.local_rank if self.env.is_gpu else self.env.device)
            targets = targets.to(self.env.local_rank if self.env.is_gpu else self.env.device)
            self._run_batch(source, targets)

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.env.local_rank == 0 and epoch % self.config.save_every == 0:
                self.checkpoint.save(self.model, epoch)

def load_train_objs(cfg: DictConfig) -> Tuple[Dataset, torch.nn.Module, torch.optim.Optimizer]:
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
        sampler=DistributedSampler(dataset)
    )

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    try:
        # Convert hydra config to TrainingConfig first
        train_config = TrainingConfig(
            max_epochs=cfg.train.total_epochs,
            save_every=cfg.train.save_every,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            snapshot_path=cfg.train.snapshot_path,
            device=cfg.train.get("device", "auto"),
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