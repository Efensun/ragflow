#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the project root directory (parent of deployment directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Function to load environment variables from .env file
load_env_file() {
    local env_file="$PROJECT_ROOT/.env"

    # Check if .env file exists
    if [ -f "$env_file" ]; then
        echo "Loading environment variables from: $env_file"
        # Source the .env file
        set -a
        source "$env_file" 
        set +a
    else
        echo "Warning: .env file not found at: $env_file"
    fi
}

# Load environment variables
load_env_file

# Set default number of workers if WS is not set or less than 1
if [[ -z "$WS" || $WS -lt 1 ]]; then
  export WS=4
fi

echo "Starting services with $WS task executors..."

# Unset HTTP proxies that might be set by Docker daemon
export http_proxy=""
export https_proxy=""
export no_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export NO_PROXY=""

# Get jemalloc path
if command -v pkg-config &> /dev/null && pkg-config --exists jemalloc; then
    export JEMALLOC_PATH=$(pkg-config --variable=libdir jemalloc)/libjemalloc.so
    echo "Using jemalloc: $JEMALLOC_PATH"
else
    export JEMALLOC_PATH=""
    echo "Warning: jemalloc not found, running without memory optimization"
fi

# Install dependencies if not already installed
if ! command -v uwsgi &> /dev/null; then
    echo "Installing uwsgi..."
    pip install uwsgi
fi

if ! command -v supervisord &> /dev/null; then
    echo "Installing supervisor..."
    pip install supervisor
fi

# Create necessary directories
sudo mkdir -p /var/log/supervisor
sudo mkdir -p /var/log/uwsgi
sudo mkdir -p /var/run/uwsgi
sudo mkdir -p /etc/supervisor/conf.d

# Copy program configuration to conf.d directory
coinex_customer_chatbot_CONF="/etc/supervisor/conf.d/coinex_customer_chatbot.conf"
echo "Copying coinex_customer_chatbot config to: $coinex_customer_chatbot_CONF"
sudo cp "$PROJECT_ROOT/deployment/coinex_customer_chatbot.conf" "$coinex_customer_chatbot_CONF"

# Update supervisor to load new configuration
echo "Reloading supervisor configuration..."
if pgrep supervisord > /dev/null; then
    echo "Supervisor is running, reloading configuration..."
    sudo supervisorctl reread
    sudo supervisorctl update
    sudo supervisorctl status
else
    echo "Starting supervisor daemon..."
    sudo supervisord
fi

echo ""
echo "=== coinex_customer_chatbot Services Started ==="
echo "Web server: http://0.0.0.0:9380"
echo "Task executors: $WS processes"
echo "Project path: $PYTHONPATH"
echo ""
echo "Management commands:"
echo "  sudo supervisorctl status              # 查看状态"
echo "  sudo supervisorctl restart coinex_customer_chatbot:*  # 重启所有coinex_customer_chatbot服务"
echo "  sudo supervisorctl logs coinex_customer_chatbot_server # 查看Web服务日志"
echo "  sudo supervisorctl logs coinex_customer_chatbot_task:coinex_customer_chatbot_task_00 # 查看任务日志" 