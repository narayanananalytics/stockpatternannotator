#!/bin/bash
# Docker entrypoint script for Stock Pattern Annotator

set -e

echo "=========================================="
echo "Stock Pattern Annotator - Starting..."
echo "=========================================="
echo ""

# Check for GPU
if [ "$DEVICE" = "cuda" ]; then
    echo "Checking GPU availability..."
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"
    echo ""
fi

# Create necessary directories
mkdir -p /data/models
mkdir -p /data/results
mkdir -p /data/databases
mkdir -p /data/tensorboard

echo "Data directories ready:"
echo "  - Models: /data/models"
echo "  - Results: /data/results"
echo "  - Databases: /data/databases"
echo "  - TensorBoard: /data/tensorboard"
echo ""

# Check database connectivity (if PostgreSQL)
if [[ $DATABASE_URL == postgresql* ]]; then
    echo "Waiting for PostgreSQL database..."
    until python -c "from sqlalchemy import create_engine; create_engine('$DATABASE_URL').connect()" 2>/dev/null; do
        echo "  PostgreSQL is unavailable - sleeping"
        sleep 2
    done
    echo "  PostgreSQL is up!"
    echo ""
fi

# Print configuration
echo "Configuration:"
echo "  Database: ${DATABASE_URL:-Not set}"
echo "  Device: ${DEVICE:-auto}"
echo "  Polygon API: ${POLYGON_API_KEY:+Configured}"
echo ""

# Execute the main command
echo "Executing: $@"
echo "=========================================="
echo ""

exec "$@"
