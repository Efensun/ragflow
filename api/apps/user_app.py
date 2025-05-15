#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import json
import re
from datetime import datetime

from flask import request, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_required, current_user, login_user, logout_user

from api.db.db_models import TenantLLM
from api.db.services.llm_service import TenantLLMService, LLMService
from api.utils.api_utils import (
    server_error_response,
    validate_request,
    get_data_error_result,
)
from api.utils import (
    get_uuid,
    get_format_time,
    decrypt,
    download_img,
    current_timestamp,
    datetime_format,
)
from api.db import UserTenantRole, FileType
from api import settings
from api.db.services.user_service import UserService, TenantService, UserTenantService
from api.db.services.file_service import FileService
from api.utils.api_utils import get_json_result, construct_response


@manager.route("/login", methods=["POST", "GET"])  # noqa: F821
def login():
    """
    User login endpoint.
    ---
    tags:
      - User
    parameters:
      - in: body
        name: body
        description: Login credentials.
        required: true
        schema:
          type: object
          properties:
            email:
              type: string
              description: User email.
            password:
              type: string
              description: User password.
    responses:
      200:
        description: Login successful.
        schema:
          type: object
      401:
        description: Authentication failed.
        schema:
          type: object
    """
    if not request.json:
        return get_json_result(
            data=False, code=settings.RetCode.AUTHENTICATION_ERROR, message="Unauthorized!"
        )

    email = request.json.get("email", "")
    users = UserService.query(email=email)
    if not users:
        return get_json_result(
            data=False,
            code=settings.RetCode.AUTHENTICATION_ERROR,
            message=f"Email: {email} is not registered!",
        )

    password = request.json.get("password")
    try:
        password = decrypt(password)
    except BaseException:
        return get_json_result(
            data=False, code=settings.RetCode.SERVER_ERROR, message="Fail to crypt password"
        )

    user = UserService.query_user(email, password)
    if user:
        response_data = user.to_json()
        user.access_token = get_uuid()
        login_user(user)
        user.update_time = (current_timestamp(),)
        user.update_date = (datetime_format(datetime.now()),)
        user.save()
        msg = "Welcome back!"
        return construct_response(data=response_data, auth=user.get_id(), message=msg)
    else:
        return get_json_result(
            data=False,
            code=settings.RetCode.AUTHENTICATION_ERROR,
            message="Email and password do not match!",
        )


@manager.route("/github_callback", methods=["GET"])  # noqa: F821
def github_callback():
    """
    GitHub OAuth callback endpoint.
    ---
    tags:
      - OAuth
    parameters:
      - in: query
        name: code
        type: string
        required: true
        description: Authorization code from GitHub.
    responses:
      200:
        description: Authentication successful.
        schema:
          type: object
    """
    import requests

    res = requests.post(
        settings.GITHUB_OAUTH.get("url"),
        data={
            "client_id": settings.GITHUB_OAUTH.get("client_id"),
            "client_secret": settings.GITHUB_OAUTH.get("secret_key"),
            "code": request.args.get("code"),
        },
        headers={"Accept": "application/json"},
    )
    res = res.json()
    if "error" in res:
        return redirect("/?error=%s" % res["error_description"])

    if "user:email" not in res["scope"].split(","):
        return redirect("/?error=user:email not in scope")

    session["access_token"] = res["access_token"]
    session["access_token_from"] = "github"
    user_info = user_info_from_github(session["access_token"])
    email_address = user_info["email"]
    users = UserService.query(email=email_address)
    user_id = get_uuid()
    if not users:
        # User isn't try to register
        try:
            try:
                avatar = download_img(user_info["avatar_url"])
            except Exception as e:
                logging.exception(e)
                avatar = ""
            users = user_register(
                user_id,
                {
                    "access_token": session["access_token"],
                    "email": email_address,
                    "avatar": avatar,
                    "nickname": user_info["login"],
                    "login_channel": "github",
                    "last_login_time": get_format_time(),
                    "is_superuser": False,
                },
            )
            if not users:
                raise Exception(f"Fail to register {email_address}.")
            if len(users) > 1:
                raise Exception(f"Same email: {email_address} exists!")

            # Try to log in
            user = users[0]
            login_user(user)
            return redirect("/?auth=%s" % user.get_id())
        except Exception as e:
            rollback_user_registration(user_id)
            logging.exception(e)
            return redirect("/?error=%s" % str(e))

    # User has already registered, try to log in
    user = users[0]
    user.access_token = get_uuid()
    login_user(user)
    user.save()
    return redirect("/?auth=%s" % user.get_id())


@manager.route("/feishu_callback", methods=["GET"])  # noqa: F821
def feishu_callback():
    """
    Feishu OAuth callback endpoint.
    ---
    tags:
      - OAuth
    parameters:
      - in: query
        name: code
        type: string
        required: true
        description: Authorization code from Feishu.
    responses:
      200:
        description: Authentication successful.
        schema:
          type: object
    """
    import requests

    app_access_token_res = requests.post(
        settings.FEISHU_OAUTH.get("app_access_token_url"),
        data=json.dumps(
            {
                "app_id": settings.FEISHU_OAUTH.get("app_id"),
                "app_secret": settings.FEISHU_OAUTH.get("app_secret"),
            }
        ),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    app_access_token_res = app_access_token_res.json()
    if app_access_token_res["code"] != 0:
        return redirect("/?error=%s" % app_access_token_res)

    res = requests.post(
        settings.FEISHU_OAUTH.get("user_access_token_url"),
        data=json.dumps(
            {
                "grant_type": settings.FEISHU_OAUTH.get("grant_type"),
                "code": request.args.get("code"),
            }
        ),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {app_access_token_res['app_access_token']}",
        },
    )
    res = res.json()
    if res["code"] != 0:
        return redirect("/?error=%s" % res["message"])

    if "contact:user.email:readonly" not in res["data"]["scope"].split():
        return redirect("/?error=contact:user.email:readonly not in scope")
    session["access_token"] = res["data"]["access_token"]
    session["access_token_from"] = "feishu"
    user_info = user_info_from_feishu(session["access_token"])
    email_address = user_info["email"]
    users = UserService.query(email=email_address)
    user_id = get_uuid()
    if not users:
        # User isn't try to register
        try:
            try:
                avatar = download_img(user_info["avatar_url"])
            except Exception as e:
                logging.exception(e)
                avatar = ""
            users = user_register(
                user_id,
                {
                    "access_token": session["access_token"],
                    "email": email_address,
                    "avatar": avatar,
                    "nickname": user_info["en_name"],
                    "login_channel": "feishu",
                    "last_login_time": get_format_time(),
                    "is_superuser": False,
                },
            )
            if not users:
                raise Exception(f"Fail to register {email_address}.")
            if len(users) > 1:
                raise Exception(f"Same email: {email_address} exists!")

            # Try to log in
            user = users[0]
            login_user(user)
            return redirect("/?auth=%s" % user.get_id())
        except Exception as e:
            rollback_user_registration(user_id)
            logging.exception(e)
            return redirect("/?error=%s" % str(e))

    # User has already registered, try to log in
    user = users[0]
    user.access_token = get_uuid()
    login_user(user)
    user.save()
    return redirect("/?auth=%s" % user.get_id())


def user_info_from_feishu(access_token):
    import requests

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {access_token}",
    }
    res = requests.get(
        "https://open.feishu.cn/open-apis/authen/v1/user_info", headers=headers
    )
    user_info = res.json()["data"]
    user_info["email"] = None if user_info.get("email") == "" else user_info["email"]
    return user_info


def user_info_from_github(access_token):
    import requests

    headers = {"Accept": "application/json", "Authorization": f"token {access_token}"}
    res = requests.get(
        f"https://api.github.com/user?access_token={access_token}", headers=headers
    )
    user_info = res.json()
    email_info = requests.get(
        f"https://api.github.com/user/emails?access_token={access_token}",
        headers=headers,
    ).json()
    user_info["email"] = next(
        (email for email in email_info if email["primary"]), None
    )["email"]
    return user_info


@manager.route("/logout", methods=["GET"])  # noqa: F821
@login_required
def log_out():
    """
    User logout endpoint.
    ---
    tags:
      - User
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Logout successful.
        schema:
          type: object
    """
    current_user.access_token = ""
    current_user.save()
    logout_user()
    return get_json_result(data=True)


@manager.route("/setting", methods=["POST"])  # noqa: F821
@login_required
def setting_user():
    """
    Update user settings.
    ---
    tags:
      - User
    security:
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        description: User settings to update.
        required: true
        schema:
          type: object
          properties:
            nickname:
              type: string
              description: New nickname.
            email:
              type: string
              description: New email.
    responses:
      200:
        description: Settings updated successfully.
        schema:
          type: object
    """
    update_dict = {}
    request_data = request.json
    if request_data.get("password"):
        new_password = request_data.get("new_password")
        if not check_password_hash(
                current_user.password, decrypt(request_data["password"])
        ):
            return get_json_result(
                data=False,
                code=settings.RetCode.AUTHENTICATION_ERROR,
                message="Password error!",
            )

        if new_password:
            update_dict["password"] = generate_password_hash(decrypt(new_password))

    for k in request_data.keys():
        if k in [
            "password",
            "new_password",
            "email",
            "status",
            "is_superuser",
            "login_channel",
            "is_anonymous",
            "is_active",
            "is_authenticated",
            "last_login_time",
        ]:
            continue
        update_dict[k] = request_data[k]

    try:
        UserService.update_by_id(current_user.id, update_dict)
        return get_json_result(data=True)
    except Exception as e:
        logging.exception(e)
        return get_json_result(
            data=False, message="Update failure!", code=settings.RetCode.EXCEPTION_ERROR
        )


@manager.route("/info", methods=["GET"])  # noqa: F821
@login_required
def user_profile():
    """
    Get user profile information.
    ---
    tags:
      - User
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: User profile retrieved successfully.
        schema:
          type: object
          properties:
            id:
              type: string
              description: User ID.
            nickname:
              type: string
              description: User nickname.
            email:
              type: string
              description: User email.
    """
    return get_json_result(data=current_user.to_dict())


def rollback_user_registration(user_id):
    try:
        UserService.delete_by_id(user_id)
    except Exception:
        pass
    try:
        TenantService.delete_by_id(user_id)
    except Exception:
        pass
    try:
        u = UserTenantService.query(tenant_id=user_id)
        if u:
            UserTenantService.delete_by_id(u[0].id)
    except Exception:
        pass
    try:
        TenantLLM.delete().where(TenantLLM.tenant_id == user_id).execute()
    except Exception:
        pass


def _create_own_tenant_resources(tenant_id_for_new_user, nickname_for_new_user, user_id_of_new_user):
    """
    辅助函数：为新用户创建其自有的租户及相关默认资源。
    """
    tenant = {
        "id": tenant_id_for_new_user,
        "name": nickname_for_new_user + "‘s Kingdom",
        "llm_id": settings.CHAT_MDL,
        "embd_id": settings.EMBEDDING_MDL,
        "asr_id": settings.ASR_MDL,
        "parser_ids": settings.PARSERS,
        "img2txt_id": settings.IMAGE2TEXT_MDL,
        "rerank_id": settings.RERANK_MDL,
        "tts_id": settings.TTS_MDL if hasattr(settings, "TTS_MDL") else "", # 添加TTS_MDL
    }
    usr_tenant_owner_link = {
        "id": get_uuid(), 
        "tenant_id": tenant_id_for_new_user,
        "user_id": user_id_of_new_user,
        "invited_by": user_id_of_new_user, 
        "role": UserTenantRole.OWNER.value,
    }
    file_id = get_uuid()
    root_file = {
        "id": file_id,
        "parent_id": file_id,
        "tenant_id": tenant_id_for_new_user,
        "created_by": user_id_of_new_user,
        "name": "/",
        "type": FileType.FOLDER.value,
        "size": 0,
        "location": "",
    }
    tenant_llm_configs = []
    if hasattr(settings, "LLM_FACTORY") and settings.LLM_FACTORY:
        for llm in LLMService.query(fid=settings.LLM_FACTORY):
            tenant_llm_configs.append(
                {
                    "tenant_id": tenant_id_for_new_user,
                    "llm_factory": settings.LLM_FACTORY,
                    "llm_name": llm.llm_name,
                    "model_type": llm.model_type,
                    "api_key": settings.API_KEY if hasattr(settings, "API_KEY") else "",
                    "api_base": settings.LLM_BASE_URL if hasattr(settings, "LLM_BASE_URL") else "",
                    "max_tokens": llm.max_tokens if llm.max_tokens else 8192
                }
            )
    
    TenantService.insert(**tenant)
    UserTenantService.insert(**usr_tenant_owner_link)
    if tenant_llm_configs:
        TenantLLMService.insert_many(tenant_llm_configs)
    FileService.insert(root_file)
    logging.info(f"Created own tenant resources for user ID {user_id_of_new_user} (Tenant ID: {tenant_id_for_new_user})")


def user_register(user_id, user_info, 
                  assign_to_existing_tenant_id=None, 
                  invited_to_existing_tenant_by_user_id=None,
                  role_in_assigned_tenant=UserTenantRole.ADMIN.value): # 新增角色参数
    """
    注册新用户。
    如果提供了 assign_to_existing_tenant_id，用户将被添加到该现有租户，
    并且不会创建其个人的租户。
    否则，将为用户创建一个新的个人租户。
    """
    # 始终先保存用户对象本身
    saved_user = UserService.save(**user_info) # user_info 应该已包含 id=user_id
    if not saved_user:
        logging.error(f"Failed to save user: {user_info.get('email')}")
        return None # 返回 None 或引发异常

    if assign_to_existing_tenant_id:
        # 将用户添加到指定的现有租户
        user_tenant_link = {
            "id": get_uuid(), 
            "tenant_id": assign_to_existing_tenant_id,
            "user_id": user_id, # 新用户的 ID
            "invited_by": invited_to_existing_tenant_by_user_id, # "拥有"目标租户的用户的ID
            "role": role_in_assigned_tenant, 
        }
        UserTenantService.insert(**user_tenant_link)
        logging.info(f"User {user_info.get('email')} (ID: {user_id}) assigned to existing tenant {assign_to_existing_tenant_id} with role {role_in_assigned_tenant}.")
    else:
        # 为用户创建新的个人租户 (租户ID与用户ID相同)
        _create_own_tenant_resources(tenant_id_for_new_user=user_id, 
                                     nickname_for_new_user=user_info["nickname"], 
                                     user_id_of_new_user=user_id)
        # logging 已在 _create_own_tenant_resources 中

    # 查询并返回刚创建或刚被引用的用户记录
    return UserService.query(email=user_info["email"])


@manager.route("/register", methods=["POST"])  # noqa: F821
@validate_request("nickname", "email", "password")
def user_add():
    """
    Register a new user.
    ---
    tags:
      - User
    parameters:
      - in: body
        name: body
        description: Registration details.
        required: true
        schema:
          type: object
          properties:
            nickname:
              type: string
              description: User nickname.
            email:
              type: string
              description: User email.
            password:
              type: string
              description: User password.
    responses:
      200:
        description: Registration successful.
        schema:
          type: object
    """

    if not settings.REGISTER_ENABLED:
        return get_json_result(
            data=False,
            message="User registration is disabled!",
            code=settings.RetCode.OPERATING_ERROR,
        )

    req = request.json
    email_address = req["email"]

    # Validate the email address
    if not re.match(r"^[\w\._-]+@([\w_-]+\.)+[\w-]{2,}$", email_address):
        return get_json_result(
            data=False,
            message=f"Invalid email address: {email_address}!",
            code=settings.RetCode.OPERATING_ERROR,
        )

    # Check if the email address is already used
    if UserService.query(email=email_address):
        return get_json_result(
            data=False,
            message=f"Email: {email_address} has already registered!",
            code=settings.RetCode.OPERATING_ERROR,
        )

    # Construct user info data
    nickname = req["nickname"]
    user_id = get_uuid() # ID for the new user
    user_dict = {
        "id": user_id, # <--- 确保 user_dict 包含 id
        "access_token": get_uuid(),
        "email": email_address,
        "nickname": nickname,
        "password": decrypt(req["password"]),
        "login_channel": "password",
        "last_login_time": get_format_time(),
        "is_superuser": False,
    }
    
    # 用于回滚的标志
    assigned_to_existing = False
    id_of_tenant_assigned_to = None

    try:
        target_email_for_tenant_source = "efen.sun@vinotech.com"
        
        if email_address.lower() == target_email_for_tenant_source.lower():
            # 如果注册用户就是目标邮件用户，则正常创建其个人租户
            logging.info(f"User {email_address} is the target tenant source; proceeding with standard personal tenant creation.")
            users = user_register(user_id, user_dict)
        else:
            # 尝试将新用户分配给目标邮件用户的租户
            source_tenant_users = UserService.query(email=target_email_for_tenant_source)
            if source_tenant_users:
                source_tenant_user = source_tenant_users[0]
                # 假设目标用户A的主要租户ID就是其用户ID
                template_tenant_id = source_tenant_user.id 
                source_tenant_owner_id = source_tenant_user.id
                
                logging.info(f"New user {email_address} will be directly assigned to tenant {template_tenant_id} from user {target_email_for_tenant_source} with OWNER role (for testing).")
                # 为新用户创建基本的User记录 (如果尚未创建或希望分离逻辑)
                # 注意：如果 user_register 也创建 UserTenant 记录，需小心重复
                # 在我们之前的方案中，user_register 内部的 _create_own_tenant_resources 不会被调用
                # 而是直接创建 UserTenant 链接。
                # 首先确保用户记录存在
                saved_user_check = UserService.query(id=user_id)
                if not saved_user_check:
                    UserService.save(**user_dict)


                UserTenantService.save(
                    id=get_uuid(),
                    tenant_id=template_tenant_id,
                    user_id=user_id, # 新用户的 ID
                    invited_by=source_tenant_owner_id,
                    role=UserTenantRole.OWNER.value # <<< 修改点：设置为 OWNER 角色 (仅用于测试)
                )
                logging.info(f"User {email_address} (ID: {user_id}) added to template tenant {template_tenant_id} with OWNER role.")
                
                # 查询新创建或引用的用户记录
                users = UserService.query(email=email_address)
                
                # 既然用户直接使用模板用户的租户作为OWNER，理论上不应该再创建自己的个人租户资源
                # 所以 _create_own_tenant_resources 不应被调用
                # 同时，原始的 user_register 函数的调用方式也需要调整，避免它创建个人租户

                # 调整：如果用户被赋予了OWNER角色到模板租户，就不再为其创建个人租户
                # （这部分逻辑在之前的方案中是，如果assign_to_existing_tenant_id则不创建个人租户）
                # 为确保清晰，我们这里假设 UserService.save 已经处理了用户创建
                # 并且 UserTenantService.save 处理了租户关联
                
                assigned_to_existing = True # 标记一下，用于回滚逻辑
                id_of_tenant_assigned_to = template_tenant_id

            else:
                # 目标邮件用户未找到，则为新用户创建个人租户 (标准流程)
                logging.warning(f"Target user {target_email_for_tenant_source} for tenant source not found. New user {email_address} will get their own personal tenant.")
                users = user_register(user_id, user_dict)

        if not users:
            # user_register 返回 None 或空列表表示失败
            raise Exception(f"User registration process failed for {email_address}.")
        
        user = users[0] # user_register 返回的是列表
        login_user(user)
        return construct_response(
            data=user.to_json(),
            auth=user.get_id(),
            message=f"{nickname}, welcome aboard!",
        )
    except Exception as e:
        logging.error(f"User registration outer exception for {email_address}: {str(e)}")
        # 精细化回滚
        if assigned_to_existing and id_of_tenant_assigned_to:
            try:
                logging.info(f"Rolling back assignment for user {user_id} from tenant {id_of_tenant_assigned_to}")
                links_to_delete = UserTenantService.query(user_id=user_id, tenant_id=id_of_tenant_assigned_to)
                for link in links_to_delete:
                    UserTenantService.delete_by_id(link.id)
                UserService.delete_by_id(user_id) 
            except Exception as rb_ex:
                logging.error(f"Error during specific rollback for assigned tenant: {str(rb_ex)}")
        else:
            logging.info(f"Rolling back standard registration for user ID {user_id}")
            rollback_user_registration(user_id) 

        logging.exception(e) 
        return get_json_result(
            data=False,
            message=f"User registration failure. Error: {str(e)}", 
            code=settings.RetCode.EXCEPTION_ERROR,
        )


@manager.route("/tenant_info", methods=["GET"])  # noqa: F821
@login_required
def tenant_info():
    """
    Get tenant information.
    ---
    tags:
      - Tenant
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Tenant information retrieved successfully.
        schema:
          type: object
          properties:
            tenant_id:
              type: string
              description: Tenant ID.
            name:
              type: string
              description: Tenant name.
            llm_id:
              type: string
              description: LLM ID.
            embd_id:
              type: string
              description: Embedding model ID.
    """
    try:
        tenants = TenantService.get_info_by(current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")
        return get_json_result(data=tenants[0])
    except Exception as e:
        return server_error_response(e)


@manager.route("/set_tenant_info", methods=["POST"])  # noqa: F821
@login_required
@validate_request("tenant_id", "asr_id", "embd_id", "img2txt_id", "llm_id")
def set_tenant_info():
    """
    Update tenant information.
    ---
    tags:
      - Tenant
    security:
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        description: Tenant information to update.
        required: true
        schema:
          type: object
          properties:
            tenant_id:
              type: string
              description: Tenant ID.
            llm_id:
              type: string
              description: LLM ID.
            embd_id:
              type: string
              description: Embedding model ID.
            asr_id:
              type: string
              description: ASR model ID.
            img2txt_id:
              type: string
              description: Image to Text model ID.
    responses:
      200:
        description: Tenant information updated successfully.
        schema:
          type: object
    """
    req = request.json
    try:
        tid = req.pop("tenant_id")
        TenantService.update_by_id(tid, req)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)
