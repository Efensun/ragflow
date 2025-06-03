

这是一个使用 uWSGI + Supervisor 部署方案，只管理两个核心服务：Web服务器和后台任务处理器。

## 架构说明

```
┌─────────────────┐
│   Supervisor    │  (进程管理器)
├─────────────────┤
│ ┌─────────────┐ │
│ │coinex_customer_chatbot_server│ │  (uWSGI + Flask Web应用)
│ │   (uWSGI)   │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │coinex_customer_chatbot_task │ │  (后台异步任务处理器)
│ │ (多进程池)   │ │  - coinex_customer_chatbot_task_00, coinex_customer_chatbot_task_01...
│ └─────────────┘ │
└─────────────────┘
```

## 文件结构

```
deployment/
├── uwsgi.ini              # uWSGI配置文件（生产环境）
├── coinex_customer_chatbot.conf           # coinex_customer_chatbot程序配置（放在 /etc/supervisor/conf.d/）
├── start.sh              # 生产环境启动脚本（使用系统supervisor）
├── local.sh              # 本地开发启动脚本（独立supervisor实例）
└── README.md            # 本文档
```

## 配置管理方式

### ✅ **标准 Supervisor 实践**
- **生产环境**: 使用系统 supervisord，配置放在 `/etc/supervisor/conf.d/coinex_customer_chatbot.conf`
- **本地开发**: 独立 supervisord 实例，完整配置文件
- **模块化管理**: 只定义程序配置，不重复系统配置

### ✅ **自动路径检测**
- 所有路径都基于项目根目录自动计算
- 不再硬编码路径
- 支持在任意位置运行

## 使用方法

### 方法1：本地开发（推荐）
```bash
# 无需sudo权限，独立supervisor实例
./deployment/local.sh
```

### 方法2：生产环境部署
```bash
# 使用系统supervisor，配置放在conf.d目录
export WS=4
./deployment/start.sh
```

### 方法3：手动部署
```bash
# 1. 安装依赖
pip install uwsgi supervisor

# 2. 复制配置到系统位置
sudo cp deployment/coinex_customer_chatbot.conf /etc/supervisor/conf.d/
export WS=4 PYTHONPATH=$(pwd) JEMALLOC_PATH=$(pkg-config --variable=libdir jemalloc)/libjemalloc.so

# 3. 重载supervisor配置
sudo supervisorctl reread
sudo supervisorctl update
```

## 主要特性

### ✅ **标准化配置管理**
- **遵循 supervisor 最佳实践**：配置文件放在 `/etc/supervisor/conf.d/`
- **模块化设计**：只包含程序配置，不重复系统配置
- **统一命名**：`coinex_customer_chatbot_server` 和 `coinex_customer_chatbot_task`

### ✅ **智能路径管理**
- **自动检测项目根目录**：无需手动配置路径
- **环境变量传递**：动态配置所有路径
- **双模式支持**：生产环境和本地开发分离

### ✅ **高性能 Web 服务**
- **uWSGI**: 生产级 WSGI 服务器
- **多进程**: 可配置worker进程数
- **内存优化**: 自动内存回收和限制

### ✅ **智能任务管理**
- **动态进程数**: 根据 `WS` 环境变量调整
- **jemalloc 优化**: 内存分配性能提升
- **自动重启**: 进程异常时自动恢复

## 管理命令

### 生产环境模式（系统supervisor）
```bash
# 查看服务状态
sudo supervisorctl status

# 重启coinex_customer_chatbot所有服务
sudo supervisorctl restart coinex_customer_chatbot:*

# 重启特定服务
sudo supervisorctl restart coinex_customer_chatbot_server
sudo supervisorctl restart coinex_customer_chatbot_task:coinex_customer_chatbot_task_00

# 查看日志
sudo supervisorctl logs coinex_customer_chatbot_server
sudo supervisorctl logs coinex_customer_chatbot_task:coinex_customer_chatbot_task_00

# 重新加载配置
sudo supervisorctl reread
sudo supervisorctl update
```

### 本地开发模式（独立supervisor）
```bash
# 在另一个终端中管理
supervisorctl -c deployment/local_supervisord.conf status
supervisorctl -c deployment/local_supervisord.conf restart coinex_customer_chatbot:*
supervisorctl -c deployment/local_supervisord.conf logs coinex_customer_chatbot_server

# 或者直接查看日志文件
tail -f logs/supervisor/coinex_customer_chatbot_server.log
tail -f logs/supervisor/coinex_customer_chatbot_task_00.log
```

## 配置文件说明

### supervisor 配置结构对比

| 模式 | 配置文件位置 | 包含内容 | 管理方式 |
|------|-------------|----------|----------|
| **生产环境** | `/etc/supervisor/conf.d/coinex_customer_chatbot.conf` | 只有程序配置 | 系统 supervisord |
| **本地开发** | `deployment/local_supervisord.conf` | 完整配置 | 独立 supervisord |

### 程序命名规范

| 程序名 | 功能 | 日志文件 |
|--------|------|----------|
| `coinex_customer_chatbot_server` | Web服务器 | `/var/log/supervisor/coinex_customer_chatbot_server.log` |
| `coinex_customer_chatbot_task` | 任务处理器 | `/var/log/supervisor/coinex_customer_chatbot_task_00.log` |

### 进程组管理

```ini
[group:coinex_customer_chatbot]
programs=coinex_customer_chatbot_server,coinex_customer_chatbot_task
```

可以通过组名操作所有服务：
```bash
sudo supervisorctl restart coinex_customer_chatbot:*
sudo supervisorctl stop coinex_customer_chatbot:*
```

## 环境变量

| 变量名 | 本地默认值 | 生产默认值 | 说明 |
|--------|------------|------------|------|
| WS | 2 | 4 | 任务执行器进程数量 |
| PYTHONPATH | 自动检测 | 自动检测 | 项目根目录路径 |
| JEMALLOC_PATH | 自动检测 | 自动检测 | jemalloc库路径 |

## 日志文件位置

### 本地开发模式
- **项目根目录**: `logs/`
- **Web服务器**: `logs/supervisor/coinex_customer_chatbot_server.log`
- **任务执行器**: `logs/supervisor/coinex_customer_chatbot_task_00.log` 等

### 生产环境模式
- **Web服务器**: `/var/log/supervisor/coinex_customer_chatbot_server.log`
- **任务执行器**: `/var/log/supervisor/coinex_customer_chatbot_task_00.log` 等

## 故障排除

### 检查配置是否正确加载
```bash
# 生产环境
sudo supervisorctl reread
sudo supervisorctl avail

# 本地开发
supervisorctl -c deployment/local_supervisord.conf avail
```

### 检查环境变量
```bash
# 检查项目路径
echo $PYTHONPATH

# 检查进程数配置
echo $WS

# 检查jemalloc路径
echo $JEMALLOC_PATH
```

### 查看详细错误信息
```bash
# 生产环境
sudo supervisorctl logs coinex_customer_chatbot_server stderr
sudo tail -f /var/log/supervisor/supervisord.log

# 本地开发
tail -f logs/supervisor/supervisord.log
```

## 配置更新流程

### 生产环境
```bash
# 1. 修改配置文件
vim deployment/coinex_customer_chatbot.conf

# 2. 复制到系统位置
sudo cp deployment/coinex_customer_chatbot.conf /etc/supervisor/conf.d/

# 3. 重载配置
sudo supervisorctl reread
sudo supervisorctl update
```

### 本地开发
```bash
# 重启即可，配置会自动重新生成
./deployment/local.sh
```


