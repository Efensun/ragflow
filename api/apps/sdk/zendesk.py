import requests
import logging
import base64
from flask import request
from api.utils.api_utils import get_result, get_error_data_result, token_required
from api import settings


def get_zendesk_config():
    """
    动态获取Zendesk配置，避免导入时机问题
    """
    return settings.ZENDESK_CONFIG or {}


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
    
    if key_id == 'your_key_id' or secret == 'your_secret':
        logging.warning("Zendesk credentials not configured properly")
        raise Exception("Zendesk credentials not configured. Please check your configuration file.")
    
    if auth_type.lower() == 'basic':
        # 使用 Basic 鉴权 (推荐方式，简单可靠)
        try:
            credentials = base64.b64encode(f"{key_id}:{secret}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
            logging.info("Using Basic authentication")
        except Exception as e:
            logging.error(f"Failed to generate Basic auth: {e}")
            raise Exception(f"Basic authentication failed: {e}")
            
    elif auth_type.lower() == 'jwt':
        # 使用 JWT 鉴权 (需要额外依赖)
        try:
            import jwt
            import time
            
            # 生成 JWT token
            payload = {
                'scope': 'app',
                'iat': int(time.time())
            }
            jwt_token = jwt.encode(payload, secret, algorithm='HS256', headers={'kid': key_id})
            headers["Authorization"] = f"Bearer {jwt_token}"
            logging.info("Using JWT authentication")
        except ImportError:
            raise Exception("JWT authentication requires 'PyJWT' package. Please install it or use 'basic' auth.")
        except Exception as e:
            logging.error(f"Failed to generate JWT token: {e}")
            raise Exception(f"JWT authentication failed: {e}")
    else:
        raise Exception(f"Unsupported auth_type: {auth_type}. Use 'basic' or 'jwt'")
    
    return headers


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
            trigger:
              type: string
              description: Event trigger type.
            messages:
              type: array
              items:
                type: object
              description: Message data.
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

        # 验证触发器类型
        if data.get('trigger') != 'message:appUser':
            return get_result(data={'status': 'ignored'})

        messages = data.get('messages', [])
        if not messages:
            return get_error_data_result(message='No messages found')

        message = messages[0]
        conversation_id = message.get('conversation', {}).get('id')
        user_message = message.get('text')
        author_id = message.get('author', {}).get('userId')

        if not all([conversation_id, user_message, author_id]):
            return get_error_data_result(message='Missing required fields: conversation_id, user_message, or author_id')

        # 处理用户消息 - 这里可以集成您的RAG系统
        response_text = process_user_message(user_message, author_id)

        # 回复用户
        success = send_reply(conversation_id, response_text)
        
        if success:
            return get_result(data={'status': 'ok', 'response_sent': True})
        else:
            return get_error_data_result(message='Failed to send response')

    except Exception as e:
        logging.exception(f"Error processing Zendesk webhook: {e}")
        return get_error_data_result(message=f'Webhook processing failed: {str(e)}')


def process_user_message(user_message: str, author_id: str) -> str:
    """
    Process user message and generate response.
    这里可以集成您的RAG系统来生成智能回复。
    
    Args:
        user_message: 用户发送的消息
        author_id: 用户ID
        
    Returns:
        str: 生成的回复消息
    """
    # TODO: 集成RAG系统
    # 示例实现：
    try:
        # 这里可以调用您的RAG检索系统
        # 例如：
        # 1. 调用知识库检索API
        # 2. 使用LLM生成回复
        # 3. 返回格式化的回复
        
        # 临时实现
        response = f"您好！我收到了您的消息：{user_message}。我正在处理中..."
        
        logging.info(f"Processed message for user {author_id}: {user_message[:50]}...")
        return response
        
    except Exception as e:
        logging.exception(f"Error processing message: {e}")
        return "抱歉，处理您的消息时出现了问题，请稍后再试。"


def send_reply(conversation_id: str, text: str) -> bool:
    """
    Send reply message to Zendesk conversation.
    根据 Smooch.io API 文档使用正确的鉴权方式
    
    Args:
        conversation_id: 对话ID
        text: 回复内容
        
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
        
        payload = {
            "author": {
                "type": "business",
                "userId": "bot"
            },
            "type": "text",
            "text": text
        }

        logging.info(f"Sending message to conversation {conversation_id} with auth type: {auth_type}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code in [200, 201]:
            logging.info(f"Successfully sent reply to conversation {conversation_id}")
            return True
        else:
            logging.error(f"Failed to send reply: {response.status_code} - {response.text}")
            logging.error(f"Request URL: {url}")
            logging.error(f"Request payload: {payload}")
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