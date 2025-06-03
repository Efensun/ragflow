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
  export WS=2  # 本地开发用较少进程
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

# Create local log directories (no sudo needed)
mkdir -p "$PROJECT_ROOT/logs/supervisor"
mkdir -p "$PROJECT_ROOT/logs/uwsgi"
mkdir -p "$PROJECT_ROOT/logs/run"

# Create local supervisor config (complete config for local development)
LOCAL_SUPERVISOR_CONF="$PROJECT_ROOT/deployment/local_supervisord.conf"
cat > "$LOCAL_SUPERVISOR_CONF" << EOF
[supervisord]
nodaemon=true
logfile=$PROJECT_ROOT/logs/supervisor/supervisord.log
pidfile=$PROJECT_ROOT/logs/run/supervisord.pid
childlogdir=$PROJECT_ROOT/logs/supervisor

[unix_http_server]
file=$PROJECT_ROOT/logs/run/supervisor.sock

[supervisorctl]
serverurl=unix://$PROJECT_ROOT/logs/run/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

; Web服务器 - 使用uWSGI
[program:coinex_customer_chatbot_server]
command=uwsgi $PROJECT_ROOT/deployment/local_uwsgi.ini
directory=$PROJECT_ROOT
autostart=true
autorestart=true
startretries=5
redirect_stderr=true
stdout_logfile=$PROJECT_ROOT/logs/supervisor/coinex_customer_chatbot_server.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=3
priority=100

; 后台任务处理器
[program:coinex_customer_chatbot_task]
command=python3 rag/svr/task_executor.py %(process_num)d
directory=$PROJECT_ROOT
process_name=%(program_name)s_%(process_num)02d
numprocs=$WS
autostart=true
autorestart=true
startretries=5
redirect_stderr=true
stdout_logfile=$PROJECT_ROOT/logs/supervisor/coinex_customer_chatbot_task_%(process_num)02d.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=3
priority=200
environment=PYTHONPATH="$PROJECT_ROOT",LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib",LD_PRELOAD="$JEMALLOC_PATH"

; 进程组配置
[group:coinex_customer_chatbot]
programs=coinex_customer_chatbot_server,coinex_customer_chatbot_task
priority=999
EOF

# Create local uWSGI config
LOCAL_UWSGI_CONF="$PROJECT_ROOT/deployment/local_uwsgi.ini"
cat > "$LOCAL_UWSGI_CONF" << EOF
[uwsgi]
# 基本配置
module = api.apps:app
chdir = $PROJECT_ROOT

# 网络配置
http = 0.0.0.0:9380
http-keepalive = 1
http-auto-chunked = 1

# 进程和线程配置
master = true
processes = 2
threads = 2
enable-threads = true

# 内存和性能优化
buffer-size = 32768
memory-report = true
max-requests = 1000
max-requests-delta = 50
reload-on-rss = 512

# 预加载应用
preload-app = true
lazy-apps = false

# 日志配置
logto = $PROJECT_ROOT/logs/uwsgi/coinex_customer_chatbot_server.log
log-maxsize = 100000000
log-backupcount = 3
log-date = %%Y-%%m-%%d %%H:%%M:%%S
disable-logging = false

# PID文件
pidfile = $PROJECT_ROOT/logs/run/coinex_customer_chatbot_server.pid

# 环境变量
env = PYTHONPATH=$PROJECT_ROOT
env = LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib
env = http_proxy=
env = https_proxy=
env = no_proxy=
env = HTTP_PROXY=
env = HTTPS_PROXY=
env = NO_PROXY=

# uWSGI内部配置
die-on-term = true
need-app = true
EOF

echo ""
echo "=== Starting coinex_customer_chatbot Local Development ==="
echo "Web server: http://0.0.0.0:9380"
echo "Task executors: $WS processes"
echo "Project path: $PYTHONPATH"
echo "Logs directory: $PROJECT_ROOT/logs/"
echo ""
echo "Management commands (in another terminal):"
echo "  supervisorctl -c $LOCAL_SUPERVISOR_CONF status"
echo "  supervisorctl -c $LOCAL_SUPERVISOR_CONF restart coinex_customer_chatbot:*"
echo "  supervisorctl -c $LOCAL_SUPERVISOR_CONF logs coinex_customer_chatbot_server"

# Start supervisor
exec supervisord -c "$LOCAL_SUPERVISOR_CONF" 