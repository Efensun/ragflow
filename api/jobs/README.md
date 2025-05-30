# 定时任务模块 (Jobs)

这个目录包含各种定时执行的任务脚本，用于数据同步、清理任务等。

## 目录结构

```
api/jobs/
├── __init__.py          # 模块初始化文件
├── zendesk_sync.py      # Zendesk数据同步脚本
├── zendesk_help_center_sync.py      # Zendesk帮助中心文档同步脚本
└── README.md           # 使用说明（本文件）
```

## Zendesk数据同步 (zendesk_sync.py)

### 功能特性

- ✅ 获取Zendesk中的所有对话数据
- ✅ 同步用户信息
- ✅ 支持增量同步（指定天数范围）
- ✅ 自动保存到本地文件系统
- ✅ 详细的日志记录
- ✅ 支持单次执行和定时执行两种模式
- ✅ 错误处理和重试机制

### 快速开始

#### 1. 安装依赖

确保已安装必要的Python包：

```bash
pip install requests schedule
```

#### 2. 配置Zendesk

确保在项目配置文件中正确设置了Zendesk的API配置：

```python
# 在 api/settings.py 中配置
ZENDESK_CONFIG = {
    'app_id': 'your_app_id',
    'key_id': 'your_key_id', 
    'secret': 'your_secret',
    'base_url': 'https://api.smooch.io',
    'auth_type': 'basic'  # 或 'bearer'
}
```

#### 3. 使用方法

**单次执行（获取最近7天数据）：**
```bash
cd /path/to/your/project
python api/jobs/zendesk_sync.py --mode once --days 7
```

**定时执行（每6小时同步一次）：**
```bash
python api/jobs/zendesk_sync.py --mode schedule --interval 6 --days 1
```

**作为Python模块使用：**
```python
from api.jobs.zendesk_sync import ZendeskDataSync

# 创建同步器实例
syncer = ZendeskDataSync()

# 执行数据同步
result = syncer.sync_all_data(days_back=7)
print(f"同步结果: {result}")
```

### 命令行参数

| 参数 | 描述 | 默认值 | 示例 |
|------|------|---------|------|
| `--mode` | 运行模式：`once`(单次) 或 `schedule`(定时) | `once` | `--mode schedule` |
| `--days` | 同步最近几天的数据 | `7` | `--days 30` |
| `--interval` | 定时执行的间隔小时数 | `6` | `--interval 12` |

### 数据存储

同步的数据会保存在以下目录结构中：

```
data/zendesk_sync/YYYYMMDD_HHMMSS/
├── conversations.json      # 对话列表
├── messages_[conv_id].json # 每个对话的消息详情
├── users.json             # 用户信息汇总
└── sync_stats.json        # 同步统计信息
```

### 日志记录

日志文件保存在：
- 文件日志：`logs/zendesk_sync_YYYYMMDD.log`
- 控制台输出：实时显示运行状态

### 使用Cron定时任务

如果要在Linux/macOS系统上设置定时任务，可以使用cron：

```bash
# 编辑crontab
crontab -e

# 添加定时任务（每6小时执行一次）
0 */6 * * * cd /path/to/your/project && python api/jobs/zendesk_sync.py --mode once --days 1 >> logs/cron_zendesk.log 2>&1
```

### 错误处理

脚本包含完善的错误处理机制：

- ✅ 网络请求超时处理
- ✅ API限流和重试
- ✅ 配置验证
- ✅ 异常捕获和日志记录
- ✅ 优雅的进程终止

### 性能优化

- 支持分页获取数据，避免内存溢出
- 请求之间添加延迟，避免API限流
- 增量同步，只获取指定时间范围的数据
- 异步处理（可选，需要额外实现）

### 故障排除

**常见问题：**

1. **配置错误**
   ```
   错误: Zendesk配置不完整，请检查配置文件
   解决: 检查 api/settings.py 中的 ZENDESK_CONFIG 配置
   ```

2. **认证失败**
   ```
   错误: 401 Unauthorized
   解决: 检查 key_id 和 secret 是否正确
   ```

3. **网络连接问题**
   ```
   错误: Cannot connect to Zendesk API
   解决: 检查网络连接和base_url设置
   ```

### 扩展功能

可以根据需要扩展以下功能：

- [ ] 数据库存储支持
- [ ] 数据清洗和格式化
- [ ] 增量同步优化
- [ ] API限流智能处理
- [ ] 监控和告警
- [ ] 数据分析和报告

### 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## Zendesk帮助中心文档同步 (zendesk_help_center_sync.py)

### 功能特性

- ✅ 获取指定语言的所有帮助中心文档（Articles）
- ✅ 同步分类（Categories）和章节（Sections）结构
- ✅ 同步文档内容（保持原始HTML格式）
- ✅ 支持多语言内容同步
- ✅ 自动保存到本地文件系统
- ✅ 详细的日志记录
- ✅ 支持单次执行和定时执行两种模式
- ✅ 错误处理和重试机制

### 快速开始

#### 1. 安装依赖

确保已安装必要的Python包：

```bash
pip install requests schedule
```

#### 2. 获取Zendesk API凭据

你需要以下信息：
- **子域名**: 你的Zendesk实例子域名（如：yourcompany）
- **邮箱**: 你的Zendesk管理员邮箱
- **API Token**: 在Zendesk管理界面中生成的API Token

获取API Token的步骤：
1. 登录Zendesk管理界面
2. 转到Admin Center > Apps and integrations > APIs > Zendesk API
3. 启用Token访问并生成新的API Token

#### 3. 使用方法

**单次执行（同步中文内容）：**
```bash
python api/jobs/zendesk_help_center_sync.py \
  --subdomain yourcompany \
  --email admin@yourcompany.com \
  --token your_api_token_here \
  --locale zh-cn
```

**同步多种语言：**
```bash
python api/jobs/zendesk_help_center_sync.py \
  --subdomain yourcompany \
  --email admin@yourcompany.com \
  --token your_api_token_here \
  --locale zh-cn \
  --locale en-us \
  --locale ja
```

**定时执行（每24小时同步一次）：**
```bash
python api/jobs/zendesk_help_center_sync.py \
  --subdomain yourcompany \
  --email admin@yourcompany.com \
  --token your_api_token_here \
  --locale zh-cn \
  --mode schedule \
  --interval 24
```

**作为Python模块使用：**
```python
from api.jobs.zendesk_help_center_sync import ZendeskHelpCenterSync

# 创建同步器实例
syncer = ZendeskHelpCenterSync(
    subdomain='yourcompany',
    email='admin@yourcompany.com', 
    api_token='your_api_token_here'
)

# 执行数据同步
result = syncer.sync_help_center_data(locales=['zh-cn', 'en-us'])
print(f"同步结果: {result}")
```

### 命令行参数

| 参数 | 描述 | 必需 | 示例 |
|------|------|---------|------|
| `--subdomain` | Zendesk子域名 | ✅ | `--subdomain yourcompany` |
| `--email` | Zendesk用户邮箱 | ✅ | `--email admin@company.com` |
| `--token` | Zendesk API Token | ✅ | `--token abc123...` |
| `--locale` | 要同步的语言代码（可重复） | ❌ | `--locale zh-cn --locale en-us` |
| `--mode` | 运行模式：`once` 或 `schedule` | ❌ | `--mode schedule` |
| `--interval` | 定时执行的间隔小时数 | ❌ | `--interval 12` |

### 支持的语言代码

常用的语言代码：
- `zh-cn` - 简体中文
- `zh-tw` - 繁体中文
- `en-us` - 英语（美式）
- `ja` - 日语
- `ko` - 韩语
- `es` - 西班牙语
- `fr` - 法语
- `de` - 德语

### 数据存储结构

同步的数据会保存在以下目录结构中：

```
data/zendesk_help_center/YYYYMMDD_HHMMSS/
├── all_help_center_data.json        # 所有语言的汇总数据
├── zh-cn_help_center_data.json      # 中文数据
├── en-us_help_center_data.json      # 英文数据
└── sync_stats.json                  # 同步统计信息
```

### 数据格式

每个语言的数据文件包含：

```json
{
  "categories": [
    {
      "id": 123,
      "name": "分类名称",
      "description": "分类描述",
      "locale": "zh-cn",
      ...
    }
  ],
  "sections": [
    {
      "id": 456,
      "name": "章节名称", 
      "description": "章节描述",
      "category_id": 123,
      "locale": "zh-cn",
      ...
    }
  ],
  "articles": [
    {
      "id": 789,
      "title": "文档标题",
      "body": "文档内容HTML（保持原始格式）",
      "section_id": 456,
      "locale": "zh-cn",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z",
      ...
    }
  ]
}
```

### 日志记录

日志文件保存在：
- 文件日志：`logs/zendesk_help_center_sync_YYYYMMDD.log`
- 控制台输出：实时显示运行状态

### 使用Cron定时任务

如果要在Linux/macOS系统上设置定时任务，可以使用cron：

```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天凌晨2点执行）
0 2 * * * cd /path/to/your/project && python api/jobs/zendesk_help_center_sync.py --subdomain yourcompany --email admin@company.com --token your_token --locale zh-cn >> logs/cron_help_center.log 2>&1
```

### 错误处理

脚本包含完善的错误处理机制：

- ✅ 网络请求超时处理
- ✅ API限流处理（自动延迟）
- ✅ 文件下载失败处理
- ✅ 认证错误检测
- ✅ 异常捕获和日志记录
- ✅ 优雅的进程终止

### 性能优化

- 支持分页获取数据，避免内存溢出
- 请求之间添加延迟，避免API限流
- 文件下载去重，避免重复下载
- 异常恢复机制

### 故障排除

**常见问题：**

1. **认证失败**
   ```
   错误: API连接测试失败: 401 Unauthorized
   解决: 检查子域名、邮箱和API Token是否正确
   ```

2. **语言不支持**
   ```
   错误: 404 Not Found
   解决: 检查指定的语言代码是否在Zendesk中启用
   ```

3. **权限不足**
   ```
   错误: 403 Forbidden
   解决: 确保API Token具有帮助中心读取权限
   ```

4. **网络连接问题**
   ```
   错误: Connection timeout
   解决: 检查网络连接和防火墙设置
   ```

### API限制

- Zendesk API有请求速率限制，脚本已自动处理
- 大量数据同步可能需要较长时间
- 附件下载受网络速度影响

### 扩展功能

可以根据需要扩展以下功能：

- [ ] 增量同步（只同步更新的内容）
- [ ] 数据库存储支持
- [ ] 内容格式转换（Markdown等）
- [ ] 全文搜索索引构建
- [ ] 监控和告警
- [ ] 自动翻译集成

### 安全注意事项

- 妥善保管API Token，不要提交到代码仓库
- 定期轮换API Token
- 使用环境变量存储敏感信息
- 限制API Token的权限范围

### 贡献

欢迎提交Issue和Pull Request来改进这个工具！ 