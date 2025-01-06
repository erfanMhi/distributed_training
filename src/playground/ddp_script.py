import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.linear(x)
        return output


class DummyDataset(Dataset):
    def __init__(self, size: int = 1000) -> None:
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def setup(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Use 'nccl' backend for GPU, 'gloo' for CPU
    if not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


def setup_logging(rank: int, log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration for both file and console output.

    Args:
        rank: The rank of the current process
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(f"ddp_trainer_{rank}")
    logger.setLevel(logging.INFO)

    # File handler - one file per rank
    fh = logging.FileHandler(str(log_path / f"ddp_rank_{rank}.log"))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Execute the training process on each distributed process.

    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup logging first
    logger = setup_logging(rank, args.log_dir)
    extra = {"rank": rank}

    # Set the same random seed for all processes
    torch.manual_seed(42)

    setup(rank, world_size, args.backend)

    model = SimpleModel()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    # Broadcast model parameters from rank 0 to all processes
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    dataset = DummyDataset()
    sampler: DistributedSampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            for name, param in model.named_parameters():
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.SUM
                )
                assert param.grad is not None  # For mypy to be happy
                param.grad /= world_size
                logger.info(
                    f"Epoch {epoch} - Parameter: {name}, "
                    f"Gradient: {torch.norm(param.grad)}",
                    extra=extra,
                )
                logger.info(
                    f"Epoch {epoch} - Parameter: {name}, "
                    f"Weight: {torch.norm(param.data)}",
                    extra=extra,
                )

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0 and rank == 0:
                logger.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}, "
                    f"Loss: {loss.item():.6f}",
                    extra=extra,
                )

        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            logger.info(
                f"Epoch: {epoch}, Average Loss: {avg_loss:.6f}", extra=extra
            )

    cleanup()


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description="PyTorch DDP Training")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of worker processes for data loading (default: 4)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="backend for distributed training (default: nccl)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=torch.cuda.device_count() if torch.cuda.is_available() else 4,
        help="number of processes for distributed training",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="directory to store log files (default: logs)",
    )
    return parser


def main() -> None:
    """Main entry point of the script."""
    parser = get_argument_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        logging.info(f"Training on {args.world_size} GPUs")
    else:
        logging.info(f"Training on CPU with {args.world_size} processes")

    mp.spawn(
        train, args=(args.world_size, args), nprocs=args.world_size, join=True
    )


if __name__ == "__main__":
    main()
