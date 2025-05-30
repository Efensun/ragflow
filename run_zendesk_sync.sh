#!/bin/bash
#
# Zendesk数据同步启动脚本
# 
# 用法:
#   ./run_zendesk_sync.sh                   # 单次执行，同步最近7天数据
#   ./run_zendesk_sync.sh --days 30         # 单次执行，同步最近30天数据  
#   ./run_zendesk_sync.sh --schedule        # 定时执行模式
#   ./run_zendesk_sync.sh --help           # 显示帮助信息
#

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Zendesk数据同步工具

用法: $0 [选项]

选项:
    --help                显示此帮助信息
    --days N             同步最近N天的数据 (默认: 7)
    --schedule           启动定时执行模式
    --interval N         定时执行间隔小时数 (默认: 6)
    --check              检查配置和依赖
    --status             显示当前运行状态

示例:
    $0                   # 单次执行，同步最近7天数据
    $0 --days 30         # 单次执行，同步最近30天数据
    $0 --schedule        # 启动定时执行模式
    $0 --check           # 检查配置和环境

EOF
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        return 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $python_version"
    return 0
}

# 检查依赖包
check_dependencies() {
    log_info "检查Python依赖包..."
    
    local missing_deps=()
    
    # 检查requests
    if ! python3 -c "import requests" 2>/dev/null; then
        missing_deps+=("requests")
    fi
    
    # 检查schedule
    if ! python3 -c "import schedule" 2>/dev/null; then
        missing_deps+=("schedule")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少依赖包: ${missing_deps[*]}"
        log_info "请运行: pip install ${missing_deps[*]}"
        return 1
    fi
    
    log_info "所有依赖包已安装"
    return 0
}

# 检查配置
check_config() {
    log_info "检查Zendesk配置..."
    
    if [ ! -f "api/settings.py" ]; then
        log_error "配置文件 api/settings.py 不存在"
        return 1
    fi
    
    # 检查配置是否包含必要字段
    if ! grep -q "ZENDESK_CONFIG" api/settings.py; then
        log_warn "配置文件中未找到 ZENDESK_CONFIG，请参考 api/jobs/config_example.py"
        return 1
    fi
    
    log_info "配置文件检查通过"
    return 0
}

# 检查环境
check_environment() {
    log_info "开始环境检查..."
    
    check_python || return 1
    check_dependencies || return 1
    check_config || return 1
    
    log_info "环境检查完成"
    return 0
}

# 检查进程状态
check_status() {
    log_info "检查Zendesk同步进程状态..."
    
    local pids=$(pgrep -f "zendesk_sync.py" 2>/dev/null)
    
    if [ -n "$pids" ]; then
        log_info "发现运行中的进程:"
        ps -p $pids -o pid,ppid,cmd --no-headers
    else
        log_info "没有运行中的Zendesk同步进程"
    fi
}

# 创建必要的目录
create_directories() {
    mkdir -p logs
    mkdir -p data/zendesk_sync
    log_debug "已创建必要的目录"
}

# 主函数
main() {
    local mode="once"
    local days=7
    local interval=6
    local check_only=false
    local status_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --days)
                days="$2"
                shift 2
                ;;
            --schedule)
                mode="schedule"
                shift
                ;;
            --interval)
                interval="$2"
                shift 2
                ;;
            --check)
                check_only=true
                shift
                ;;
            --status)
                status_only=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示状态
    if [ "$status_only" = true ]; then
        check_status
        exit 0
    fi
    
    # 仅检查环境
    if [ "$check_only" = true ]; then
        check_environment
        exit $?
    fi
    
    # 检查环境
    if ! check_environment; then
        log_error "环境检查失败，请修复后重试"
        exit 1
    fi
    
    # 创建必要目录
    create_directories
    
    # 构建命令
    local cmd="python3 api/jobs/zendesk_sync.py --mode $mode --days $days"
    
    if [ "$mode" = "schedule" ]; then
        cmd="$cmd --interval $interval"
    fi
    
    # 显示执行信息
    log_info "准备执行Zendesk数据同步"
    log_info "模式: $mode"
    log_info "同步天数: $days"
    if [ "$mode" = "schedule" ]; then
        log_info "执行间隔: $interval 小时"
    fi
    
    # 执行命令
    log_info "开始执行: $cmd"
    exec $cmd
}

# 脚本入口
main "$@" 