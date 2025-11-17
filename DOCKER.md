# Docker Quick Start Guide

Complete guide for running Stock Pattern Annotator in Docker with CPU and GPU support.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) 2.0+
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (for GPU only)

## Quick Start (TL;DR)

```bash
# Clone and setup
git clone https://github.com/narayanananalytics/stockpatternannotator.git
cd stockpatternannotator
make setup

# Edit .env with your settings
nano .env

# Run with GPU (recommended for 16GB GPU)
make build-gpu
make up-gpu

# Or run with CPU
make build-cpu
make up-cpu

# Check GPU
make gpu-check

# Run RL training
make example-rl-gpu
```

## Detailed Setup

### 1. Initial Setup

```bash
# Create data directories and .env file
make setup

# Edit .env with your API keys
nano .env
```

Required `.env` settings:
```bash
# Database (SQLite or PostgreSQL)
DATABASE_URL=sqlite:////data/databases/stockpatterns.db

# Polygon.io API key (get from https://polygon.io)
POLYGON_API_KEY=your_api_key_here

# GPU settings (if using GPU)
NVIDIA_VISIBLE_DEVICES=0
DEVICE=cuda
```

### 2. Build Docker Images

#### CPU Version

```bash
make build-cpu
# Or manually:
# docker build -t stockpatternannotator:cpu .
```

#### GPU Version (16GB GPU Optimized)

```bash
make build-gpu
# Or manually:
# docker build --build-arg CUDA_VERSION=11.8.0 -t stockpatternannotator:gpu .
```

### 3. Verify Installation

```bash
# Check GPU availability (GPU only)
make gpu-check

# Should show:
# ✓ CUDA Available: YES
# Device Name: Your GPU Model
# Total Memory: 16.00 GB
```

## Running Examples

### Fetch Data from Polygon.io

```bash
# First, add your POLYGON_API_KEY to .env

# Fetch OHLC data
docker-compose run --rm app-cpu python examples/polygon_pipeline_example.py
```

### Validate Patterns

```bash
docker-compose run --rm app-cpu python examples/pattern_validation_example.py
```

### Train RL Agent

#### GPU (Recommended - 8-12x faster)

```bash
# Full pipeline with GPU
make example-rl-gpu

# Custom timesteps (500k for better results)
docker-compose run --rm -e TOTAL_TIMESTEPS=500000 app-gpu \
    python examples/rl_trading_example.py
```

Expected output:
```
======================================================================
GPU INFORMATION
======================================================================
✓ CUDA Available: YES
  GPU Memory: 16.0 GB
  Optimized n_steps: 8192
  Optimized batch_size: 512

Training Time Estimate: ~5-10 minutes for 500K timesteps
```

#### CPU (Slower)

```bash
make example-rl-cpu

# Or manually
docker-compose run --rm app-cpu python examples/rl_trading_example.py
```

## Services

### PostgreSQL Database (Optional)

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Update .env to use PostgreSQL
DATABASE_URL=postgresql://stockpattern:stockpattern123@postgres:5432/stockpatterns

# Connect to database
docker-compose exec postgres psql -U stockpattern -d stockpatterns
```

### TensorBoard Monitoring

```bash
# Start TensorBoard
make tensorboard
# Or: docker-compose --profile monitoring up -d tensorboard

# Access at http://localhost:6006
```

### Jupyter Notebook (Development)

```bash
# Start Jupyter
make up-dev
# Or: docker-compose --profile development up -d jupyter

# Access at http://localhost:8888
```

## Common Workflows

### Development Workflow

```bash
# 1. Setup
make setup
nano .env  # Add API keys

# 2. Start services
make up-gpu
make tensorboard

# 3. Fetch and prepare data
docker-compose exec app-gpu python examples/polygon_pipeline_example.py
docker-compose exec app-gpu python examples/pattern_validation_example.py

# 4. Train with monitoring
docker-compose exec app-gpu python examples/rl_trading_example.py

# 5. View results in TensorBoard
# Visit http://localhost:6006

# 6. Analyze results
make up-dev
# Visit http://localhost:8888
```

### Production Workflow

```bash
# 1. Use PostgreSQL for database
docker-compose up -d postgres

# 2. Update .env
DATABASE_URL=postgresql://stockpattern:stockpattern123@postgres:5432/stockpatterns

# 3. Run training pipeline
docker-compose --profile gpu up -d

# 4. Check logs
docker-compose logs -f app-gpu

# 5. Results saved to ./data/models and ./data/results
```

## Data Persistence

All data persists in the `./data` directory:

```
./data/
├── models/
│   ├── pattern_trading_agent.zip      # Latest trained model
│   └── best_model/                    # Best model during training
├── results/
│   └── backtest.csv                   # Detailed backtest results
├── databases/
│   └── stockpatterns.db              # SQLite database
└── tensorboard/
    └── PPO_*/                         # TensorBoard logs
```

Backup your data:
```bash
tar -czf stockpattern-data-$(date +%Y%m%d).tar.gz ./data
```

## GPU Optimization

### Verify GPU Access

```bash
# 1. Check host GPU
nvidia-smi

# 2. Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 3. Check in container
make gpu-check
```

### GPU Memory Management

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# If out of memory, reduce batch size
docker-compose run --rm app-gpu python -c "
from stockpatternannotator import RLPipeline
pipeline = RLPipeline()
# ... load data ...
pipeline.train_agent(
    total_timesteps=100000,
    hyperparameters={'batch_size': 256, 'n_steps': 4096}
)
"
```

### Performance Tuning for 16GB GPU

```python
# examples/custom_training.py
from stockpatternannotator import RLPipeline

pipeline = RLPipeline(
    database_url='sqlite:////data/databases/stockpatterns.db'
)

pipeline.load_data()
pipeline.validate_patterns()
pipeline.calculate_features()
pipeline.prepare_environments()

# Optimized for 16GB GPU
pipeline.train_agent(
    total_timesteps=1000000,  # 1M timesteps in ~10-15 min
    hyperparameters={
        'batch_size': 512,      # Large batch for stable gradients
        'n_steps': 8192,        # Large rollout buffer
        'learning_rate': 3e-4,
        'n_epochs': 10
    },
    save_path='/data/models/optimized_agent'
)
```

## Troubleshooting

### GPU Issues

**Error: "CUDA not available"**
```bash
# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-docker:
# https://github.com/NVIDIA/nvidia-docker
```

**Error: "Out of GPU memory"**
```bash
# Reduce batch size
hyperparameters = {'batch_size': 256, 'n_steps': 4096}

# Or close other GPU applications
nvidia-smi  # Check what's using GPU
```

### Database Issues

**Error: "Connection refused" (PostgreSQL)**
```bash
# Wait for PostgreSQL to start
docker-compose up -d postgres
sleep 5

# Check status
docker-compose ps postgres
docker-compose logs postgres
```

**Error: "Database locked" (SQLite)**
```bash
# SQLite doesn't support concurrent writes well
# Either:
# 1. Use PostgreSQL for production
# 2. Don't run multiple containers accessing same SQLite DB
```

### Port Conflicts

**Error: "Port already in use"**
```bash
# Change ports in .env
TENSORBOARD_PORT=6007
JUPYTER_PORT=8889
POSTGRES_PORT=5433

# Restart services
docker-compose down
docker-compose up -d
```

### Container Issues

**Container exits immediately**
```bash
# Check logs
docker-compose logs app-gpu

# Run interactively to debug
docker-compose run --rm app-gpu bash
```

**Slow builds**
```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
make build-gpu
```

## Cleanup

```bash
# Stop all services
make down

# Remove all containers and volumes
make clean

# Remove Docker images
docker rmi stockpatternannotator:cpu stockpatternannotator:gpu

# Full cleanup (careful - removes all data!)
docker-compose down -v
rm -rf ./data
```

## Advanced Usage

### Custom Entrypoint

```bash
# Run custom Python script
docker-compose run --rm app-gpu python my_script.py

# Interactive Python
docker-compose run --rm app-gpu python

# Bash shell
docker-compose run --rm app-gpu bash
```

### Environment Variables

```bash
# Override environment variables
docker-compose run --rm \
    -e TOTAL_TIMESTEPS=1000000 \
    -e POLYGON_API_KEY=your_key \
    app-gpu python examples/rl_trading_example.py
```

### Volume Mounts

```bash
# Mount additional directories
docker-compose run --rm \
    -v $(pwd)/my_data:/data/custom \
    app-gpu python my_script.py
```

### Multi-GPU Training

```bash
# Use specific GPU
NVIDIA_VISIBLE_DEVICES=0 docker-compose --profile gpu up

# Use multiple GPUs (requires code changes for multi-GPU training)
NVIDIA_VISIBLE_DEVICES=0,1 docker-compose --profile gpu up
```

## Best Practices

1. **Use GPU for training** - 8-12x faster
2. **Use PostgreSQL for production** - Better concurrency than SQLite
3. **Monitor with TensorBoard** - Track training progress
4. **Backup ./data regularly** - Contains all models and results
5. **Use .env for configuration** - Don't commit secrets
6. **Start small, scale up** - Test with 50K timesteps, then increase
7. **Check GPU memory** - Monitor with `nvidia-smi`
8. **Use docker-compose profiles** - Only run services you need

## Support

- GitHub Issues: https://github.com/narayanananalytics/stockpatternannotator/issues
- Docker Documentation: https://docs.docker.com
- nvidia-docker: https://github.com/NVIDIA/nvidia-docker
