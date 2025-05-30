import requests
import logging
import base64
from flask import request
import uuid
import json
import re  # 用于正则表达式处理markdown图片

from api.utils.api_utils import get_result, get_error_data_result, token_required
from api import settings
from api.db import StatusEnum


def get_zendesk_config():
    """
    动态获取Zendesk配置，避免导入时机问题
    """
    return settings.ZENDESK_CONFIG or {}


def convert_markdown_images_to_links(text: str) -> str:
    """
    将markdown图片语法转换为链接形式，以便在Zendesk中正确显示
    
    Args:
        text: 包含markdown图片语法的文本
        
    Returns:
        str: 转换后的文本，图片变为可点击链接
    """
    try:
        # 正则表达式匹配markdown图片语法：![alt text](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        def replace_image(match):
            alt_text = match.group(1) or "查看图片"  # 如果没有alt文本，使用默认文本
            image_url = match.group(2)
            
            # 转换为链接形式，添加图片emoji作为视觉提示
            return f"📷 [{alt_text}]({image_url})"
        
        # 执行替换
        converted_text = re.sub(image_pattern, replace_image, text)
        
        # 如果发生了转换，记录日志
        if converted_text != text:
            image_count = len(re.findall(image_pattern, text))
            logging.info(f"Converted {image_count} markdown images to clickable links")
            logging.debug(f"Original text: {text[:200]}...")
            logging.debug(f"Converted text: {converted_text[:200]}...")
        
        return converted_text
        
    except Exception as e:
        logging.error(f"Error converting markdown images to links: {e}")
        return text  # 出错时返回原文本


def get_auth_headers():
    """
    根据配置生成正确的鉴权头
    
    Returns:
        dict: 包含鉴权信息的请求头
    """
    headers = {"Content-Type": "application/json"}
    
    # 动态获取配置
    zendesk_config = get_zendesk_config()
    app_id = zendesk_config.get('app_id', 'your_app_id')
    key_id = zendesk_config.get('key_id', 'your_key_id')
    secret = zendesk_config.get('secret', 'your_secret')
    auth_type = zendesk_config.get('auth_type', 'basic')
    
    # 检查配置是否完整
    if not zendesk_config:
        logging.warning("Zendesk configuration not found, using default values")
    
    if not all([app_id, key_id, secret]):
        logging.warning("Zendesk credentials not configured properly")
        return headers
    
    # 根据认证类型设置Authorization头
    if auth_type.lower() == 'basic':
        # Basic Authentication
        auth_string = f"{key_id}:{secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        headers['Authorization'] = f'Basic {auth_b64}'
        logging.info("Using Basic authentication")
    elif auth_type.lower() == 'bearer':
        # Bearer Token Authentication  
        headers['Authorization'] = f'Bearer {secret}'
        logging.info("Using Bearer authentication")
    else:
        logging.warning(f"Unknown auth_type: {auth_type}, falling back to Basic")
        auth_string = f"{key_id}:{secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        headers['Authorization'] = f'Basic {auth_b64}'
    
    return headers


def call_ragflow_assistant(user_message: str, user_id: str, conversation_id: str) -> str:
    """
    调用RAGFlow助理API获取智能回复，支持session管理保持对话上下文
    
    Args:
        user_message: 用户发送的消息
        user_id: 用户ID
        conversation_id: Zendesk对话ID（用于session管理）
        
    Returns:
        str: 助理生成的回复
    """
    try:
        # 获取配置
        zendesk_config = get_zendesk_config()
        assistant_id = zendesk_config.get('assistant_id')
        api_token = zendesk_config.get('api_token')
        
        if not assistant_id or not api_token:
            logging.error("RAGFlow assistant_id or api_token not configured")
            return "抱歉，聊天助理配置不完整，请联系管理员。"
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        
        api_base = "http://localhost:9380"  # RAGFlow默认地址
        
        # Step 1: 获取或创建session
        session_id = get_or_create_session(api_base, headers, assistant_id, conversation_id, user_id)
        
        if not session_id:
            logging.error("Failed to get or create session")
            return "抱歉，无法建立对话会话，请稍后再试。"
        
        # Step 2: 使用session进行对话
        url = f"{api_base}/api/v1/chats/{assistant_id}/completions"
        
        payload = {
            "question": user_message,
            "session_id": session_id,  # 使用session保持上下文
            "stream": False,
            "quote": True,
            "user_id": user_id
        }
        
        logging.info(f"Calling RAGFlow assistant {assistant_id} with session {session_id} for user {user_id}")
        logging.debug(f"Request payload: {payload}")
        
        # 发送请求
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        logging.info(f"Response status: {response.status_code}")
        logging.info(f"Response headers: {dict(response.headers)}")
        logging.info(f"Response text: {response.text[:500]}...")  # 显示前500字符
        
        if response.status_code == 200:
            try:
                result = response.json()
                logging.info(f"Parsed JSON type: {type(result)}")
                logging.debug(f"RAGFlow response: {result}")
                
                if isinstance(result, dict) and result.get('code') == 0:
                    data = result.get('data', {})
                    if isinstance(data, dict):
                        answer = data.get('answer', '')
                    else:
                        # 如果data不是字典，可能是流式响应的格式
                        logging.warning(f"Unexpected data format: {type(data)}, content: {data}")
                        answer = str(data) if data else "抱歉，获取回复时出现格式问题。"
                    
                    if answer:
                        logging.info(f"Got answer from RAGFlow: {answer}")
                        return answer
                    else:
                        logging.warning("Empty answer from RAGFlow")
                        return "抱歉，我没有找到相关的回答。"
                else:
                    error_msg = result.get('message', 'Unknown error') if isinstance(result, dict) else str(result)
                    logging.error(f"RAGFlow API error: {error_msg}")
                    return f"抱歉，处理您的问题时出现错误：{error_msg}"
                    
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                logging.error(f"Raw response: {response.text}")
                return "抱歉，服务器响应格式异常，请稍后再试。"
            except Exception as e:
                logging.exception(f"Error processing response: {e}")
                return "抱歉，处理服务器响应时出现问题。"
        else:
            logging.error(f"RAGFlow API request failed: {response.status_code} - {response.text}")
            return "抱歉，当前服务繁忙，请稍后再试。"
            
    except requests.exceptions.Timeout:
        logging.error("RAGFlow API request timeout")
        return "抱歉，处理时间过长，请稍后再试。"
    except requests.exceptions.ConnectionError:
        logging.error("Cannot connect to RAGFlow API")
        return "抱歉，无法连接到聊天服务，请检查网络连接。"
    except Exception as e:
        logging.exception(f"Error calling RAGFlow assistant: {e}")
        return "抱歉，处理您的消息时出现了问题，请稍后再试。"


def get_or_create_session(api_base: str, headers: dict, assistant_id: str, conversation_id: str, user_id: str) -> str:
    """
    获取或创建RAGFlow会话
    使用Zendesk conversation_id作为session名称，实现会话复用
    
    Args:
        api_base: RAGFlow API基础地址
        headers: 请求头
        assistant_id: 助理ID
        conversation_id: Zendesk对话ID
        user_id: 用户ID
        
    Returns:
        str: session_id，失败返回None
    """
    try:
        # 使用conversation_id作为session名称，便于识别和管理
        session_name = f"zendesk_{conversation_id}"
        
        # Step 1: 查找现有session
        list_url = f"{api_base}/api/v1/chats/{assistant_id}/sessions"
        list_params = {
            "name": session_name,
            "page": 1,
            "page_size": 10
        }
        
        response = requests.get(list_url, headers=headers, params=list_params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 0:
                sessions = result.get('data', [])
                if sessions:
                    # 找到现有session，直接使用
                    session_id = sessions[0]['id']
                    logging.info(f"Found existing session: {session_id} for conversation {conversation_id}")
                    return session_id
        
        # Step 2: 创建新session
        create_url = f"{api_base}/api/v1/chats/{assistant_id}/sessions"
        create_payload = {
            "name": session_name,
            "user_id": user_id
        }
        
        response = requests.post(create_url, json=create_payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 0:
                session_id = result.get('data', {}).get('id')
                logging.info(f"Created new session: {session_id} for conversation {conversation_id}")
                return session_id
            else:
                logging.error(f"Failed to create session: {result.get('message')}")
        else:
            logging.error(f"Session creation failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logging.exception(f"Error managing session: {e}")
    
    return None


@manager.route('/zendesk/webhook', methods=['POST'])  # noqa: F821
def handle_webhook():
    """
    Handle Zendesk webhook callbacks.
    ---
    tags:
      - Zendesk Integration
    parameters:
      - in: body
        name: body
        description: Webhook payload from Zendesk.
        required: true
        schema:
          type: object
          properties:
            events:
              type: array
              items:
                type: object
              description: Event data from Zendesk.
    responses:
      200:
        description: Webhook processed successfully.
        schema:
          type: object
          properties:
            status:
              type: string
              description: Processing status.
      400:
        description: Bad request.
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message.
    """
    try:
        data = request.json
        logging.info(f"Zendesk webhook received: {data}")

        # 检查是否有事件数据
        events = data.get('events', [])
        if not events:
            return get_result(data={'status': 'ignored', 'reason': 'no events'})

        # 处理第一个事件
        event = events[0]
        event_type = event.get('type')
        
        # 只处理对话消息事件
        if event_type != 'conversation:message':
            return get_result(data={'status': 'ignored', 'reason': f'event type: {event_type}'})

        # 获取消息数据
        payload = event.get('payload', {})
        conversation = payload.get('conversation', {})
        message = payload.get('message', {})
        
        conversation_id = conversation.get('id')
        user_message = message.get('content', {}).get('text')
        author = message.get('author', {})
        author_id = author.get('userId')
        author_type = author.get('type')
        
        # 只处理用户发送的消息，忽略机器人自己的消息
        if author_type != 'user':
            return get_result(data={'status': 'ignored', 'reason': f'author type: {author_type}'})

        if not all([conversation_id, user_message, author_id]):
            logging.warning(f"Missing required fields: conversation_id={conversation_id}, user_message={user_message}, author_id={author_id}")
            return get_error_data_result(message='Missing required fields: conversation_id, user_message, or author_id')

        logging.info(f"Processing message from user {author_id} in conversation {conversation_id}: {user_message}")

        # 处理用户消息 - 这里可以集成您的RAG系统
        response_text = call_ragflow_assistant(user_message, author_id, conversation_id)

        # 回复用户
        success = send_reply(conversation_id, response_text)
        
        if success:
            return get_result(data={'status': 'ok', 'response_sent': True, 'conversation_id': conversation_id})
        else:
            return get_error_data_result(message='Failed to send response')

    except Exception as e:
        logging.exception(f"Error processing Zendesk webhook: {e}")
        return get_error_data_result(message=f'Webhook processing failed: {str(e)}')


def send_reply(conversation_id: str, text: str) -> bool:
    """
    Send reply message to Zendesk conversation.
    将markdown图片转换为链接形式，并使用markdownText发送
    
    Args:
        conversation_id: 对话ID
        text: 回复内容（markdown格式）
        
    Returns:
        bool: 发送是否成功
    """
    try:
        # 动态获取配置
        zendesk_config = get_zendesk_config()
        app_id = zendesk_config.get('app_id', 'your_app_id')
        base_url = zendesk_config.get('base_url', 'https://api.smooch.io')
        auth_type = zendesk_config.get('auth_type', 'basic')
        
        # 正确构建完整的 API URL
        url = f"{base_url}/v2/apps/{app_id}/conversations/{conversation_id}/messages"
        
        # 使用正确的鉴权方式
        headers = get_auth_headers()
        
        # 将markdown图片转换为链接形式
        processed_text = convert_markdown_images_to_links(text)
        
        payload = {
            "author": {
                "type": "business",
                "userId": "bot"
            },
            "content": {
                "type": "text",
                "markdownText": processed_text  # 使用markdownText发送处理后的内容
            }
        }

        logging.info(f"Sending markdown message to conversation {conversation_id} with auth type: {auth_type}")
        logging.debug(f"Final payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code in [200, 201]:
            logging.info(f"Successfully sent reply to conversation {conversation_id}")
            return True
        else:
            logging.error(f"Failed to send reply: {response.status_code} - {response.text}")
            logging.error(f"Request URL: {url}")
            logging.error(f"Request payload: {json.dumps(payload, indent=2)}")
            return False
            
    except Exception as e:
        logging.exception(f"Error sending reply: {e}")
        return False


@manager.route('/zendesk/test', methods=['POST'])  # noqa: F821
@token_required
def test_integration(tenant_id):
    """
    Test Zendesk integration.
    ---
    tags:
      - Zendesk Integration
    security:
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        description: Test parameters.
        required: true
        schema:
          type: object
          properties:
            conversation_id:
              type: string
              description: Test conversation ID.
            message:
              type: string
              description: Test message.
    responses:
      200:
        description: Test successful.
        schema:
          type: object
          properties:
            status:
              type: string
              description: Test result.
    """
    try:
        req = request.json
        conversation_id = req.get('conversation_id')
        test_message = req.get('message', 'This is a test message from RAGFlow integration.')
        
        if not conversation_id:
            return get_error_data_result(message='conversation_id is required')
        
        success = send_reply(conversation_id, test_message)
        
        # 动态获取配置
        zendesk_config = get_zendesk_config()
        auth_type = zendesk_config.get('auth_type', 'basic')
        
        if success:
            return get_result(data={
                'status': 'success',
                'message': 'Test message sent successfully',
                'conversation_id': conversation_id,
                'auth_type': auth_type
            })
        else:
            return get_error_data_result(message='Failed to send test message')
            
    except Exception as e:
        logging.exception(f"Error in test integration: {e}")
        return get_error_data_result(message=f'Test failed: {str(e)}')


@manager.route('/zendesk/config', methods=['GET'])  # noqa: F821
@token_required
def get_config(tenant_id):
    """
    Get current Zendesk configuration (without sensitive data).
    ---
    tags:
      - Zendesk Integration
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Configuration retrieved successfully.
        schema:
          type: object
          properties:
            app_id:
              type: string
              description: Smooch App ID.
            base_url:
              type: string
              description: Smooch API base URL.
            auth_type:
              type: string
              description: Authentication type (jwt/basic).
    """
    try:
        # 动态获取配置
        zendesk_config = get_zendesk_config()
        app_id = zendesk_config.get('app_id', 'your_app_id')
        base_url = zendesk_config.get('base_url', 'https://api.smooch.io')
        auth_type = zendesk_config.get('auth_type', 'basic')
        key_id = zendesk_config.get('key_id', 'your_key_id')
        secret = zendesk_config.get('secret', 'your_secret')
        
        config = {
            'app_id': app_id,
            'base_url': base_url,
            'auth_type': auth_type,
            'key_id': key_id if key_id != 'your_key_id' else 'Not configured',
            'secret_configured': bool(secret and secret != 'your_secret')
        }
        
        return get_result(data=config)
        
    except Exception as e:
        logging.exception(f"Error getting config: {e}")
        return get_error_data_result(message=f'Failed to get config: {str(e)}')


@manager.route('/zendesk/test-auth', methods=['GET'])  # noqa: F821
@token_required
def test_auth(tenant_id):
    """
    Test authentication with Smooch.io API.
    验证当前的鉴权配置是否正确
    ---
    tags:
      - Zendesk Integration
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Authentication test result.
        schema:
          type: object
          properties:
            status:
              type: string
              description: Test result.
            auth_config:
              type: object
              description: Current auth configuration (without secrets).
    """
    try:
        # 动态获取配置
        zendesk_config = get_zendesk_config()
        app_id = zendesk_config.get('app_id', 'your_app_id')
        base_url = zendesk_config.get('base_url', 'https://api.smooch.io')
        auth_type = zendesk_config.get('auth_type', 'basic')
        key_id = zendesk_config.get('key_id', 'your_key_id')
        secret = zendesk_config.get('secret', 'your_secret')
        
        # 测试鉴权配置
        headers = get_auth_headers()
        test_url = f"{base_url}/v2/apps/{app_id}"
        
        logging.info(f"Testing auth with URL: {test_url}")
        logging.info(f"Auth type: {auth_type}")
        logging.info(f"Headers (without auth): {{'Content-Type': headers.get('Content-Type')}}")
        
        response = requests.get(test_url, headers=headers, timeout=30)
        
        result = {
            'status': 'success' if response.status_code == 200 else 'failed',
            'status_code': response.status_code,
            'auth_type': auth_type,
            'base_url': base_url,
            'app_id': app_id,
            'key_id_configured': bool(key_id and key_id != 'your_key_id'),
            'secret_configured': bool(secret and secret != 'your_secret'),
        }
        
        if response.status_code == 200:
            result['message'] = 'Authentication successful'
            result['app_info'] = response.json()
        else:
            result['message'] = f'Authentication failed: {response.text}'
            result['error_details'] = {
                'status_code': response.status_code,
                'response_text': response.text[:200]  # 只显示前200字符
            }
        
        return get_result(data=result)
        
    except Exception as e:
        logging.exception(f"Error testing auth: {e}")
        return get_error_data_result(message=f'Auth test failed: {str(e)}')


@manager.route('/zendesk/setup', methods=['GET'])  # noqa: F821
@token_required
def setup_assistant(tenant_id):
    """
    获取助理设置信息
    ---
    tags:
      - Zendesk Integration
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Setup information retrieved successfully.
        schema:
          type: object
          properties:
            assistants:
              type: array
              description: 可用的助理列表
            tokens:
              type: array
              description: 可用的API令牌
            current_config:
              type: object
              description: 当前Zendesk配置
    """
    try:
        from api.db.services.dialog_service import DialogService
        from api.db.services.api_service import APITokenService
        from api.db.services.user_service import UserTenantService
        
        # 获取当前租户的助理列表
        assistants = DialogService.query(tenant_id=tenant_id, status=StatusEnum.VALID.value)
        assistant_list = []
        for assistant in assistants:
            assistant_dict = assistant.to_json()
            assistant_list.append({
                'id': assistant_dict['id'],
                'name': assistant_dict['name'],
                'description': assistant_dict.get('description', ''),
                'create_time': assistant_dict.get('create_time', '')
            })
        
        # 获取API令牌列表
        tokens = APITokenService.query(tenant_id=tenant_id)
        token_list = []
        for token in tokens:
            token_dict = token.to_dict()
            # 只显示token的前缀和后缀，中间用*号隐藏
            masked_token = f"{token_dict['token'][:12]}***{token_dict['token'][-8:]}" if len(token_dict['token']) > 20 else token_dict['token']
            token_list.append({
                'dialog_id': token_dict['dialog_id'],
                'dialog_name': token_dict.get('dialog_name', ''),
                'token': masked_token,
                'full_token': token_dict['token'],  # 用于配置
                'source': token_dict.get('source', 'chat'),
                'tenant_id': token_dict['tenant_id']
            })
        
        # 获取当前配置
        zendesk_config = get_zendesk_config()
        current_config = {
            'assistant_id': zendesk_config.get('assistant_id', ''),
            'api_token': zendesk_config.get('api_token', ''),
            'configured': bool(zendesk_config.get('assistant_id') and zendesk_config.get('api_token'))
        }
        
        return get_result(data={
            'assistants': assistant_list,
            'tokens': token_list,
            'current_config': current_config,
            'help': {
                'step1': '从上面的assistants列表中选择一个助理ID',
                'step2': '从上面的tokens列表中选择对应的API令牌，或创建新的令牌',
                'step3': '将assistant_id和api_token配置到conf/service_conf.yaml的zendesk部分',
                'example_config': {
                    'assistant_id': 'your_assistant_id_here',
                    'api_token': 'ragflow-your_api_token_here'
                }
            }
        })
        
    except Exception as e:
        logging.exception(f"Error getting setup info: {e}")
        return get_error_data_result(message=f'Setup failed: {str(e)}')


@manager.route('/zendesk/create-token', methods=['POST'])  # noqa: F821
@token_required  
def create_api_token(tenant_id):
    """
    为指定助理创建API令牌
    ---
    tags:
      - Zendesk Integration
    security:
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        description: API token creation parameters.
        required: true
        schema:
          type: object
          properties:
            assistant_id:
              type: string
              description: 助理ID
    responses:
      200:
        description: API token created successfully.
        schema:
          type: object
          properties:
            token:
              type: string
              description: 新创建的API令牌
    """
    try:
        from api.db.services.dialog_service import DialogService
        from api.db.services.api_service import APITokenService
        
        req = request.json
        assistant_id = req.get('assistant_id')
        
        if not assistant_id:
            return get_error_data_result(message='assistant_id is required')
        
        # 验证助理是否存在且属于当前租户
        assistant = DialogService.query(tenant_id=tenant_id, id=assistant_id, status=StatusEnum.VALID.value)
        if not assistant:
            return get_error_data_result(message='Assistant not found or not accessible')
        
        # 创建API令牌
        from api.utils import get_uuid
        import secrets
        
        token = f"ragflow-{secrets.token_urlsafe(32)}"
        
        # 保存到数据库
        api_token_data = {
            'token': token,
            'dialog_id': assistant_id,
            'tenant_id': tenant_id,
            'source': 'zendesk'  # 标记为Zendesk集成使用
        }
        
        existing_token = APITokenService.query(tenant_id=tenant_id, dialog_id=assistant_id)
        if existing_token:
            # 更新现有令牌
            APITokenService.update_by_id(existing_token[0].id, api_token_data)
            logging.info(f"Updated API token for assistant {assistant_id}")
        else:
            # 创建新令牌
            APITokenService.save(**api_token_data)
            logging.info(f"Created new API token for assistant {assistant_id}")
        
        return get_result(data={
            'token': token,
            'assistant_id': assistant_id,
            'message': 'API token created successfully',
            'config_instruction': f'Add the following to your zendesk config:\nassistant_id: {assistant_id}\napi_token: {token}'
        })
        
    except Exception as e:
        logging.exception(f"Error creating API token: {e}")
        return get_error_data_result(message=f'Token creation failed: {str(e)}') 