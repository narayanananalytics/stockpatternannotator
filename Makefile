# Stock Pattern Annotator - Docker Makefile
# Quick commands for Docker operations

.PHONY: help build-cpu build-gpu up-cpu up-gpu down logs shell test clean

# Default target
help:
	@echo "Stock Pattern Annotator - Docker Commands"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build-cpu          Build CPU-only Docker image"
	@echo "  make build-gpu          Build GPU-enabled Docker image"
	@echo ""
	@echo "Run Commands:"
	@echo "  make up-cpu             Start CPU services"
	@echo "  make up-gpu             Start GPU services (requires nvidia-docker)"
	@echo "  make up-dev             Start development environment with Jupyter"
	@echo "  make up-monitoring      Start with TensorBoard monitoring"
	@echo ""
	@echo "Management Commands:"
	@echo "  make down               Stop all services"
	@echo "  make logs               View logs"
	@echo "  make shell-cpu          Open shell in CPU container"
	@echo "  make shell-gpu          Open shell in GPU container"
	@echo "  make tensorboard        Start TensorBoard on port 6006"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean              Remove containers and volumes"
	@echo "  make test               Run tests in container"
	@echo "  make gpu-check          Check GPU availability in container"

# Build images
build-cpu:
	@echo "Building CPU image..."
	docker build -t stockpatternannotator:cpu .

build-gpu:
	@echo "Building GPU image..."
	docker build --build-arg CUDA_VERSION=11.8.0 -t stockpatternannotator:gpu .

# Run services
up-cpu:
	@echo "Starting CPU services..."
	docker-compose --profile cpu up -d

up-gpu:
	@echo "Starting GPU services..."
	docker-compose --profile gpu up -d

up-dev:
	@echo "Starting development environment..."
	docker-compose --profile cpu --profile development up -d

up-monitoring:
	@echo "Starting with monitoring..."
	docker-compose --profile cpu --profile monitoring up -d

# Stop services
down:
	@echo "Stopping all services..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f

logs-app:
	docker-compose logs -f app-cpu app-gpu

logs-db:
	docker-compose logs -f postgres

# Shell access
shell-cpu:
	docker-compose run --rm app-cpu bash

shell-gpu:
	docker-compose run --rm app-gpu bash

# TensorBoard
tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	docker-compose --profile monitoring up -d tensorboard

# Testing
test:
	docker-compose run --rm app-cpu python -m pytest tests/

# GPU check
gpu-check:
	docker-compose run --rm app-gpu python -c "from stockpatternannotator.rl_gpu_utils import print_gpu_info; print_gpu_info()"

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	docker system prune -f

# Example workflows
example-polygon:
	docker-compose run --rm app-cpu python examples/polygon_pipeline_example.py

example-validation:
	docker-compose run --rm app-cpu python examples/pattern_validation_example.py

example-rl-cpu:
	docker-compose run --rm app-cpu python examples/rl_trading_example.py

example-rl-gpu:
	docker-compose run --rm app-gpu python examples/rl_trading_example.py

# Create data directories
setup:
	@echo "Creating data directories..."
	mkdir -p data/models data/results data/databases data/tensorboard
	@echo "Copying environment template..."
	cp -n .env.example .env || true
	@echo "Setup complete! Edit .env with your settings."
