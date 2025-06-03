#!/bin/bash


set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
DOMAIN_NAME="your-domain.com"  # 请修改为您的域名
BACKEND_HOST="127.0.0.1:9380"  # 后端服务地址
WEB_ROOT="/var/www/coinex_customer_chatbot"    # 网站根目录
NGINX_SITE_NAME="coinex_customer_chatbot"      # Nginx站点名称
PROJECT_DIR=$(pwd)             # 当前项目目录

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以root权限运行
check_root_privileges() {
    if [ "$EUID" -eq 0 ]; then
        log_warning "检测到以root权限运行，建议使用sudo执行此脚本"
    fi
}

# 检查系统依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js未安装，请先安装Node.js (版本 >= 18.20.4)"
        exit 1
    fi
    
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    REQUIRED_VERSION="18.20.4"
    if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
        log_error "Node.js版本过低，当前版本: $NODE_VERSION，需要版本 >= $REQUIRED_VERSION"
        exit 1
    fi
    log_success "Node.js版本检查通过: $NODE_VERSION"
    
    # 检查npm
    if ! command -v npm &> /dev/null; then
        log_error "npm未安装"
        exit 1
    fi
    log_success "npm检查通过"
    
    # 检查nginx
    if ! command -v nginx &> /dev/null; then
        log_error "Nginx未安装，正在尝试安装..."
        sudo apt update
        sudo apt install -y nginx
        log_success "Nginx安装完成"
    else
        log_success "Nginx检查通过"
    fi
}

# 获取用户配置
get_user_config() {
    log_info "配置部署参数..."
    
    read -p "请输入域名 (默认: localhost): " input_domain
    DOMAIN_NAME=${input_domain:-"localhost"}
    
    read -p "请输入后端服务地址 (默认: 127.0.0.1:9380): " input_backend
    BACKEND_HOST=${input_backend:-"127.0.0.1:9380"}
    
    read -p "是否启用HTTPS? (y/N): " enable_https
    ENABLE_HTTPS=${enable_https:-"n"}
    
    if [[ $ENABLE_HTTPS =~ ^[Yy]$ ]]; then
        read -p "请输入SSL证书路径: " SSL_CERT_PATH
        read -p "请输入SSL私钥路径: " SSL_KEY_PATH
        
        if [[ ! -f "$SSL_CERT_PATH" ]] || [[ ! -f "$SSL_KEY_PATH" ]]; then
            log_error "SSL证书或私钥文件不存在"
            exit 1
        fi
    fi
    
    log_success "配置参数获取完成"
}

# 构建前端
build_frontend() {
    log_info "开始构建前端..."
    
    if [[ ! -d "web" ]]; then
        log_error "未找到web目录，请确保在正确的项目根目录执行此脚本"
        exit 1
    fi
    
    cd web
    
    # 安装依赖
    log_info "安装前端依赖..."
    npm install
    
    # 执行构建
    log_info "执行前端构建..."
    npm run build
    
    # 检查构建结果
    if [[ ! -d "dist" ]]; then
        log_error "前端构建失败，未找到dist目录"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log_success "前端构建完成"
}

# 创建Nginx配置文件
create_nginx_config() {
    log_info "创建Nginx配置文件..."
    
    if [[ $ENABLE_HTTPS =~ ^[Yy]$ ]]; then
        # HTTPS配置
        cat > "/tmp/${NGINX_SITE_NAME}" << EOF
# HTTP重定向到HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN_NAME};
    return 301 https://\$server_name\$request_uri;
}

# HTTPS配置
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ${DOMAIN_NAME};
    root ${WEB_ROOT};
    index index.html;

    # SSL证书配置
    ssl_certificate ${SSL_CERT_PATH};
    ssl_certificate_key ${SSL_KEY_PATH};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # 安全头设置
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # 启用gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        image/svg+xml;

    # 静态资源缓存策略
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        access_log off;
    }

    # Monaco编辑器资源
    location /vs/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # API代理配置
    location /api/ {
        proxy_pass http://${BACKEND_HOST};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket支持
    location /v1/ {
        proxy_pass http://${BACKEND_HOST};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # SPA路由支持
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # 禁止访问隐藏文件
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }

    # 错误页面
    error_page 404 /index.html;
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }

    # 日志配置
    access_log /var/log/nginx/${NGINX_SITE_NAME}_access.log;
    error_log /var/log/nginx/${NGINX_SITE_NAME}_error.log;
}
EOF
    else
        # HTTP配置
        cat > "/tmp/${NGINX_SITE_NAME}" << EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN_NAME};
    root ${WEB_ROOT};
    index index.html;

    # 启用gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        image/svg+xml;

    # 静态资源缓存策略
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        access_log off;
    }

    # Monaco编辑器资源
    location /vs/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # API代理配置
    location /api/ {
        proxy_pass http://${BACKEND_HOST};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket支持
    location /v1/ {
        proxy_pass http://${BACKEND_HOST};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # SPA路由支持
    location / {
        try_files \$uri \$uri/ /index.html;
        
        # 安全头设置
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    }

    # 禁止访问隐藏文件
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }

    # 错误页面
    error_page 404 /index.html;
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }

    # 日志配置
    access_log /var/log/nginx/${NGINX_SITE_NAME}_access.log;
    error_log /var/log/nginx/${NGINX_SITE_NAME}_error.log;
}
EOF
    fi
    
    log_success "Nginx配置文件创建完成"
}

# 备份现有部署
backup_existing_deployment() {
    if [[ -d "$WEB_ROOT" ]]; then
        BACKUP_DIR="${WEB_ROOT}.backup.$(date +%Y%m%d_%H%M%S)"
        log_info "备份现有部署到: $BACKUP_DIR"
        sudo cp -r "$WEB_ROOT" "$BACKUP_DIR"
        log_success "备份完成"
    fi
}

# 部署文件
deploy_files() {
    log_info "部署前端文件..."
    
    # 创建网站根目录
    sudo mkdir -p "$WEB_ROOT"
    
    # 复制构建好的文件
    sudo cp -r web/dist/* "$WEB_ROOT/"
    
    # 设置权限
    sudo chown -R www-data:www-data "$WEB_ROOT"
    sudo chmod -R 755 "$WEB_ROOT"
    
    log_success "前端文件部署完成"
}

# 配置Nginx
configure_nginx() {
    log_info "配置Nginx..."
    
    # 移动配置文件到正确位置
    sudo mv "/tmp/${NGINX_SITE_NAME}" "/etc/nginx/sites-available/${NGINX_SITE_NAME}"
    
    # 禁用默认站点
    if [[ -L "/etc/nginx/sites-enabled/default" ]]; then
        sudo rm /etc/nginx/sites-enabled/default
        log_info "已禁用默认Nginx站点"
    fi
    
    # 启用新站点
    sudo ln -sf "/etc/nginx/sites-available/${NGINX_SITE_NAME}" "/etc/nginx/sites-enabled/${NGINX_SITE_NAME}"
    
    # 测试配置
    log_info "测试Nginx配置..."
    if sudo nginx -t; then
        log_success "Nginx配置测试通过"
    else
        log_error "Nginx配置测试失败"
        exit 1
    fi
    
    # 重新加载Nginx
    log_info "重新加载Nginx..."
    sudo systemctl reload nginx
    
    # 确保Nginx已启动
    sudo systemctl enable nginx
    sudo systemctl start nginx
    
    log_success "Nginx配置完成"
}

# 验证部署
verify_deployment() {
    log_info "验证部署..."
    
    # 检查Nginx状态
    if sudo systemctl is-active --quiet nginx; then
        log_success "Nginx服务运行正常"
    else
        log_error "Nginx服务未运行"
        exit 1
    fi
    
    # 检查端口监听
    if sudo netstat -tlnp | grep -q ":80\|:443"; then
        log_success "端口监听正常"
    else
        log_warning "端口监听检查异常，请手动检查"
    fi
    
    # 测试HTTP响应
    local test_url="http://localhost"
    if [[ $ENABLE_HTTPS =~ ^[Yy]$ ]]; then
        test_url="https://localhost"
    fi
    
    log_info "测试服务响应..."
    if curl -s -o /dev/null -w "%{http_code}" "$test_url" | grep -q "200\|301\|302"; then
        log_success "服务响应正常"
    else
        log_warning "服务响应测试异常，请手动检查"
    fi
}

# 显示部署结果
show_deployment_info() {
    log_success "🎉 coinex_customer_chatbot前端服务部署完成！"
    echo ""
    echo "=================================================="
    echo "部署信息:"
    echo "  域名: $DOMAIN_NAME"
    echo "  网站根目录: $WEB_ROOT"
    echo "  后端服务: $BACKEND_HOST"
    if [[ $ENABLE_HTTPS =~ ^[Yy]$ ]]; then
        echo "  访问地址: https://$DOMAIN_NAME"
        echo "  SSL已启用"
    else
        echo "  访问地址: http://$DOMAIN_NAME"
    fi
    echo "=================================================="
    echo ""
    echo "有用的命令:"
    echo "  查看Nginx状态: sudo systemctl status nginx"
    echo "  查看访问日志: sudo tail -f /var/log/nginx/${NGINX_SITE_NAME}_access.log"
    echo "  查看错误日志: sudo tail -f /var/log/nginx/${NGINX_SITE_NAME}_error.log"
    echo "  重新加载Nginx: sudo systemctl reload nginx"
    echo ""
    echo "如需回滚，备份文件位于: ${WEB_ROOT}.backup.*"
}

# 主函数
main() {
    echo "=================================================="
    echo "coinex_customer_chatbot前端服务Nginx自动化部署脚本"
    echo "=================================================="
    echo ""
    
    check_root_privileges
    check_dependencies
    get_user_config
    build_frontend
    create_nginx_config
    backup_existing_deployment
    deploy_files
    configure_nginx
    verify_deployment
    show_deployment_info
}

# 错误处理
trap 'log_error "部署过程中出现错误，退出码: $?"; exit 1' ERR

# 执行主函数
main "$@"