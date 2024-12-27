<div align="center">

# 🚀 PyTorch Distributed Training from Ground Up

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-package-blueviolet)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive guide and implementation for understanding distributed training in PyTorch - from low-level primitives to production deployment.

[Getting Started](#-quick-start) • [Features](#-learning-path) • [Documentation](#-implementation-details) • [Contributing](#-contributing)

</div>

---

## 🎯 Introduction

This project serves as an educational resource for understanding distributed training in PyTorch, implementing both DistributedDataParallel (DDP) and Fully Sharded Data Parallel (FSDP) from scratch using PyTorch primitives. What sets this repository apart is the complete infrastructure setup guide alongside the training implementation, bridging the gap between theoretical understanding and practical deployment.

## 📚 Learning Path

### 1. Distributed Training Fundamentals
- Implementation of DDP from scratch using PyTorch primitives
- Understanding data parallelism and gradient synchronization
- Process group management and initialization
- (Coming Soon) FSDP implementation and memory optimization

### 2. Infrastructure Setup
- Complete Terraform configurations for GPU cluster deployment
- Automated node discovery and coordination
- Network configuration for distributed training
- Shared filesystem setup for checkpointing

## 🚀 Quick Start

<details>
<summary><b>Prerequisites</b></summary>

- Python ≥ 3.10
- Poetry for dependency management
- Terraform for infrastructure setup
- Access to Nebius Cloud (infrastructure code can be adapted for other providers)
</details>

### 🖥 Local Development

1. Clone and setup:
```bash
git clone https://github.com/erfanMhi/distributed_training.git
cd distributed_training
poetry install
```

2. Run training on multiple GPUs:
```bash
poetry run torchrun \
    --nproc_per_node=NUM_GPUS \
    src/multigpu_multi_node.py EPOCHS SAVE_FREQUENCY \
    --batch_size BATCH_SIZE
```

### Cloud Deployment

1. Set up cloud credentials:
```bash
export NB_AUTHKEY_PRIVATE_PATH="path/to/private/key"
export NB_AUTHKEY_PUBLIC_ID="your-public-key-id"
export NB_SA_ID="your-service-account-id"
```

> **⚠️ Important:** For multi-node training on Nebius Cloud, ensure you have sufficient quota allocation. You'll need quota for at least 2 GPU nodes in your target region. Check your quotas in the Nebius Cloud Console and request increases if needed before deployment.

2. Deploy infrastructure:
```bash
cd infrastructure
terraform init
terraform apply
```

## 📂 Project Structure

<details open>
<summary><b>Repository Layout</b></summary>

```
distributed_training/
├── src/                      # Training implementation
│   ├── multigpu_multi_node.py  # DDP training script
│   └── data_utils.py           # Dataset utilities
├── infrastructure/           # Cloud deployment code
│   ├── main.tf              # Main Terraform configuration
│   ├── variables.tf         # Infrastructure variables
│   └── scripts/             # Deployment scripts
└── docs/                    # (Coming Soon) Detailed documentation
```
</details>

## ⚙️ Configuration

<details>
<summary><b>Training Parameters</b></summary>

| Parameter | Description | Default |
|:----------|:------------|:---------|
| `total_epochs` | Number of training epochs | - |
| `save_every` | Checkpoint frequency | - |
| `batch_size` | Batch size per GPU | 32 |

</details>

<details>
<summary><b>Infrastructure Parameters</b></summary>

| Parameter | Description | Default |
|:----------|:------------|:---------|
| `cluster_size` | Number of nodes | 1 |
| `training_epochs` | Total epochs | 10 |
| `save_frequency` | Checkpoint frequency | 5 |

</details>

## 📖 Implementation Details

### Distributed Training
- Process group initialization and management
- Gradient synchronization across nodes
- Efficient data loading with DistributedSampler
- Checkpoint management for fault tolerance

### Infrastructure
- H100 GPU cluster orchestration
- Inter-node networking setup
- Shared filesystem configuration
- Automatic training coordination

## 🛣 Roadmap

<div align="center">

| Status | Feature |
|:------:|:--------|
| ✅ | Basic DDP implementation |
| ✅ | Multi-node training support |
| ✅ | Infrastructure automation |
| 🚧 | FSDP implementation |
| 📝 | Performance optimization guides |
| 🎯 | Multi-cloud support |

</div>

## 🤝 Contributing

We welcome contributions of all kinds! Here's how you can help:

- 📝 Improve documentation
- 🐛 Report or fix bugs
- ✨ Propose or implement new features
- 🚀 Optimize performance

Please feel free to submit issues and pull requests.

## 📬 Contact

<div align="center">

[![Email](https://img.shields.io/badge/Email-mhi.erfan1%40gmail.com-blue?style=flat-square&logo=gmail)](mailto:mhi.erfan1@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-erfanMhi-black?style=flat-square&logo=github)](https://github.com/erfanMhi/distributed_training)

</div>

---

<div align="center">

If you find this project helpful, please consider giving it a ⭐!

<a href="https://github.com/erfanMhi/distributed_training">
  <img src="https://img.shields.io/github/stars/erfanMhi/distributed_training?style=social" alt="GitHub stars">
</a>

</div>
