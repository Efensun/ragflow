import hashlib
import json
import os
import time
from datetime import datetime, timedelta
import requests
import schedule
from api.db import FileType, TaskStatus
from api.db.db_models import Task
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.task_service import TaskService, queue_tasks
from api.utils import get_uuid
from api.utils.file_utils import filename_type
from rag.utils.storage_factory import STORAGE_IMPL
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


logger.info("=" * 50)
logger.info("ClickUp文档同步脚本启动")
logger.info("=" * 50)

COINEX_WEB_FOLDER_ID = ''
COINEX_PRODUCT_FOLDER_ID = ''

# 配置信息
CLICKUP_TOKEN = os.getenv('CLICKUP_TOKEN')
WORKSPACE_ID = '9008230771'
TENANT_ID = os.getenv('RAGFLOW_TENANT_ID', 'default_tenant')
LAST_SYNC_TIME_FILE = './clickup_last_sync.json'

RAGFLOW_WEB_PARENT_FOLDER_ID = 'f905d88a65ff11f09a61cf9d445194dc'  # 技术文档的父文件夹ID
RAGFLOW_PRODUCT_PARENT_FOLDER_ID = 'ffc7036065ff11f09a61cf9d445194dc'  # 产品文档的父文件夹ID

RAGFLOW_WEB_KB_ID = '345887da660e11f09a61cf9d445194dc'  # Web文档知识库ID
RAGFLOW_PRODUCT_KB_ID = '8076260e660e11f09a61cf9d445194dc'  # 产品文档知识库ID


def save_last_sync_time(timestamp):
    """保存最后同步时间"""
    try:
        with open(LAST_SYNC_TIME_FILE, 'w') as f:
            json.dump({'last_sync': timestamp}, f)
        logger.debug(f"保存同步时间成功: {timestamp}")
    except Exception as e:
        logger.error(f"保存同步时间失败: {e}")


def get_last_sync_time():
    """获取最后同步时间"""
    try:
        if os.path.exists(LAST_SYNC_TIME_FILE):
            with open(LAST_SYNC_TIME_FILE, 'r') as f:
                data = json.load(f)
                return data.get('last_sync')
    except Exception as e:
        logger.error(f"读取同步时间失败: {e}")

    # 默认返回7天前的时间戳（毫秒）
    default_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    logger.warning("使用默认同步时间（7天前）")
    return default_time


def get_all_pages(parent_id):
    """获取指定文件夹下的所有ClickUp文档页面"""
    url = f"https://api.clickup.com/api/v3/workspaces/{WORKSPACE_ID}/docs"
    params = {
        "deleted": "false",
        "archived": "false",
        "parent_id": parent_id,
        "parent_type": "FOLDER",
        "limit": 50
    }

    headers = {
        "accept": "application/json",
        "Authorization": CLICKUP_TOKEN
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        logger.debug(f"成功获取ClickUp页面，parent_id: {parent_id}")
        return response.json()
    except Exception as e:
        logger.error(f"获取ClickUp页面失败: {e}")
        return None


def get_doc_content(doc_id):
    """获取ClickUp文档内容"""
    url = f"https://api.clickup.com/api/v3/workspaces/{WORKSPACE_ID}/docs/{doc_id}/pages?max_page_depth=-1&content_format=text%2Fmd"

    headers = {
        "accept": "application/json",
        "Authorization": CLICKUP_TOKEN
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logger.debug(f"成功获取文档内容，doc_id: {doc_id}")
        return response.json()
    except Exception as e:
        logger.error(f"获取文档内容失败 {doc_id}: {e}")
        return None


def should_sync_doc(doc_created_time, last_sync_time):
    """判断文档是否需要同步"""
    try:
        # ClickUp的时间戳可能是字符串格式，需要转换
        if isinstance(doc_created_time, str):
            # 尝试解析ISO格式时间
            try:
                dt = datetime.fromisoformat(doc_created_time.replace('Z', '+00:00'))
                doc_timestamp = int(dt.timestamp() * 1000)
            except:
                # 如果是时间戳字符串
                doc_timestamp = int(doc_created_time)
        else:
            doc_timestamp = int(doc_created_time)

        return doc_timestamp > last_sync_time

    except Exception as e:
        logger.error(f"解析文档创建时间失败: {e}, doc_created_time: {doc_created_time}")
        # 如果解析失败，默认同步
        return True


def process_doc_content(doc_data, parent_name="", last_sync_time=None):
    """递归处理文档内容，提取需要同步的页面"""
    documents = []

    def extract_content(doc, prefix=""):
        doc_name = doc.get('name', '')
        doc_content = doc.get('content', '')
        doc_created = doc.get('date_created')
        doc_updated = doc.get('date_updated')

        # 检查是否需要同步（基于创建时间或更新时间）
        should_sync = False
        if last_sync_time:
            # 检查创建时间
            if doc_created and should_sync_doc(doc_created, last_sync_time):
                should_sync = True
            # 如果创建时间不满足，检查更新时间
            elif doc_updated and should_sync_doc(doc_updated, last_sync_time):
                should_sync = True
        else:
            # 如果没有指定同步时间，则同步所有文档
            should_sync = True

        if doc_name and doc_content and should_sync:
            # 创建完整的文档名称
            full_name = f"{prefix}{doc_name}" if prefix else doc_name
            if parent_name:
                full_name = f"{parent_name}_{full_name}"

            documents.append({
                'name': full_name,
                'content': doc_content,
                'date_created': doc_created,
                'date_updated': doc_updated
            })

            logger.info(f"文档需要同步: {full_name}, 创建时间: {doc_created}")
        elif doc_name and not should_sync:
            logger.debug(f"文档跳过同步: {doc_name}, 创建时间: {doc_created}")

        # 处理子页面
        if doc.get('pages'):
            for page in doc['pages']:
                extract_content(page, f"{doc_name}_" if doc_name else "")

    if isinstance(doc_data, list):
        for doc in doc_data:
            extract_content(doc)
    else:
        extract_content(doc_data)

    return documents


def bind_file_to_kb(file_id, kb_id):
    """将文件绑定到知识库"""
    try:
        # 检查知识库是否存在
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        if not e:
            logger.error(f"知识库不存在: {kb_id}")
            return False, "知识库不存在"

        # 检查文件是否存在
        e, file = FileService.get_by_id(file_id)
        if not e:
            logger.error(f"文件不存在: {file_id}")
            return False, "文件不存在"

        # 检查是否已经绑定
        existing_bindings = File2DocumentService.get_by_file_id(file_id)
        for binding in existing_bindings:
            e, doc = DocumentService.get_by_id(binding.document_id)
            if e and doc.kb_id == kb_id:
                logger.info(f"文件 {file.name} 已绑定到知识库 {kb.name}")
                return True, "已绑定"

        # 删除现有绑定
        for binding in existing_bindings:
            doc_id = binding.document_id
            e, doc = DocumentService.get_by_id(doc_id)
            if e:
                tenant_id = DocumentService.get_tenant_id(doc_id)
                if tenant_id:
                    DocumentService.remove_document(doc, tenant_id)
        File2DocumentService.delete_by_file_id(file_id)

        # 创建新的文档记录
        doc = DocumentService.insert({
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": FileService.get_parser(file.type, file.name, kb.parser_id),
            "parser_config": kb.parser_config,
            "created_by": TENANT_ID,
            "type": file.type,
            "name": file.name,
            "location": file.location,
            "size": file.size
        })

        # 创建文件到文档的绑定
        file2document = File2DocumentService.insert({
            "id": get_uuid(),
            "file_id": file_id,
            "document_id": doc.id,
        })

        logger.info(f"成功将文件 {file.name} 绑定到知识库 {kb.name}")
        return True, "绑定成功"

    except Exception as e:
        logger.error(f"绑定文件到知识库失败: {e}")
        return False, str(e)


def compare_file_content(file_id, new_content):
    """比较文件内容是否发生变化"""
    try:
        e, file = FileService.get_by_id(file_id)
        if not e:
            return True  # 如果文件不存在，认为内容有变化

        # 获取现有文件内容
        existing_blob = STORAGE_IMPL.get(file.parent_id, file.location)
        if not existing_blob:
            return True  # 如果无法获取现有内容，认为有变化

        new_content_bytes = new_content.encode('utf-8')

        # 比较内容大小和哈希
        if len(existing_blob) != len(new_content_bytes):
            return True

        existing_hash = hashlib.md5(existing_blob).hexdigest()
        new_hash = hashlib.md5(new_content_bytes).hexdigest()

        return existing_hash != new_hash

    except Exception as e:
        logger.error(f"比较文件内容失败: {e}")
        return True  # 出错时认为有变化，安全起见


def update_existing_file(file_id, new_content, new_name=None):
    """更新现有文件的内容并重新加入处理队列"""
    try:
        e, file = FileService.get_by_id(file_id)
        if not e:
            return False, "文件不存在"

        # 准备新内容
        blob = new_content.encode('utf-8')

        # 更新存储中的文件内容
        STORAGE_IMPL.put(file.parent_id, file.location, blob)

        # 更新文件记录
        update_data = {
            "size": len(blob),
            "update_time": int(time.time() * 1000)
        }

        if new_name and new_name != file.name:
            update_data["name"] = new_name

        success = FileService.update_by_id(file_id, update_data)
        if not success:
            return False, "更新文件记录失败"

        # 重置关联文档的处理状态，并重新加入队列
        file_to_docs = File2DocumentService.get_by_file_id(file_id)
        for f2d in file_to_docs:
            doc_id = f2d.document_id
            e, doc = DocumentService.get_by_id(doc_id)
            if not e:
                continue

            # 清除现有的索引数据和任务
            try:
                from rag.nlp import search
                from api import settings

                tenant_id = DocumentService.get_tenant_id(doc_id)
                if tenant_id:
                    # 删除现有任务
                    TaskService.filter_delete([Task.doc_id == doc_id])

                    # 清除索引数据
                    if settings.docStoreConn.indexExist(search.index_name(tenant_id), doc.kb_id):
                        settings.docStoreConn.delete({"doc_id": doc_id}, search.index_name(tenant_id), doc.kb_id)

            except Exception as index_e:
                logger.error(f"清除索引数据失败: {index_e}")

            # 重置文档状态
            doc_update = {
                "run": TaskStatus.RUNNING.value,  # 直接设为运行状态
                "progress": 0,
                "progress_msg": "文档已更新，开始重新处理",
                "chunk_num": 0,
                "token_num": 0
            }
            DocumentService.update_by_id(doc_id, doc_update)

            # 重新加入处理队列
            try:
                # 准备文档信息
                doc_dict = doc.to_dict()
                doc_dict["tenant_id"] = tenant_id

                # 获取存储地址
                bucket, name = File2DocumentService.get_storage_address(doc_id=doc_id)

                # 加入处理队列
                queue_tasks(doc_dict, bucket, name, 0)

                logger.info(f"文档 {doc.name} 已重新加入处理队列")

            except Exception as queue_e:
                logger.error(f"重新加入队列失败: {queue_e}")
                # 如果加入队列失败，将状态重置为未开始
                DocumentService.update_by_id(doc_id, {
                    "run": "0",
                    "progress": 0,
                    "progress_msg": "文档已更新，但加入队列失败，请手动启动处理"
                })

        logger.info(f"成功更新文件内容: {file.name}")
        return True, "文件内容更新成功并已重新加入处理队列"

    except Exception as e:
        logger.error(f"更新文件内容失败: {e}")
        return False, str(e)


def upload_doc_to_ragflow(doc_content, doc_name, parent_folder_id, kb_id):
    """上传文档到RAGFlow文件服务的指定文件夹并绑定到知识库"""
    try:
        # 检查父文件夹ID和知识库ID是否设置
        if not parent_folder_id:
            logger.warning(f"父文件夹ID未设置，跳过上传文档: {doc_name}")
            return False, "父文件夹ID未设置"

        if not kb_id:
            logger.warning(f"知识库ID未设置，跳过上传文档: {doc_name}")
            return False, "知识库ID未设置"

        # 准备文件名
        filename = f"{doc_name}.md"

        # 检查文件是否已存在
        existing_files = FileService.query(name=filename, parent_id=parent_folder_id)

        if existing_files:
            logger.info(f"文档 {filename} 已存在，检查内容是否有变化")
            existing_file = existing_files[0]  # 取第一个匹配的文件

            # 比较文件内容
            content_changed = compare_file_content(existing_file.id, doc_content)

            if not content_changed:
                logger.info(f"文档 {filename} 内容未变化，跳过更新")

                # 检查是否已绑定到指定知识库
                existing_bindings = File2DocumentService.get_by_file_id(existing_file.id)
                is_bound_to_kb = False
                for binding in existing_bindings:
                    e, doc = DocumentService.get_by_id(binding.document_id)
                    if e and doc.kb_id == kb_id:
                        is_bound_to_kb = True
                        break

                if not is_bound_to_kb:
                    # 如果未绑定到指定知识库，进行绑定
                    success, message = bind_file_to_kb(existing_file.id, kb_id)
                    if success:
                        return True, "文档已存在，完成绑定"
                    else:
                        return False, f"文档已存在，绑定失败: {message}"

                return True, "文档已存在且内容未变化"

            else:
                logger.info(f"文档 {filename} 内容有变化，更新现有文档")

                # 更新现有文件内容
                success, message = update_existing_file(existing_file.id, doc_content, filename)
                if success:
                    # 确保绑定到知识库
                    bind_success, bind_message = bind_file_to_kb(existing_file.id, kb_id)
                    if bind_success:
                        return True, "文档内容已更新且已绑定"
                    else:
                        return False, f"文档内容更新成功但绑定失败: {bind_message}"
                else:
                    return False, f"更新文档内容失败: {message}"

        # 文件不存在，创建新文件
        logger.info(f"创建新文档: {filename}")

        # 确定文件类型
        filetype = filename_type(filename)
        if not filetype:
            filetype = FileType.DOC.value

        # 生成唯一的存储位置名称
        location = filename
        while STORAGE_IMPL.obj_exist(parent_folder_id, location):
            location += "_"

        # 将 Markdown 内容转换为字节
        blob = doc_content.encode('utf-8')

        # 创建文件记录（不使用duplicate_name，直接使用原始文件名）
        file_data = {
            "id": get_uuid(),
            "parent_id": parent_folder_id,
            "tenant_id": TENANT_ID,
            "created_by": TENANT_ID,
            "type": filetype,
            "name": filename,  # 直接使用原始文件名
            "location": location,
            "size": len(blob),
            "source_type": "clickup"  # 标记来源为 ClickUp
        }

        # 插入文件记录到数据库
        file_record = FileService.insert(file_data)

        # 存储文件内容到存储系统
        STORAGE_IMPL.put(parent_folder_id, location, blob)

        logger.warning(f"成功上传新文档到文件服务: {filename}")

        # 绑定文件到知识库
        success, message = bind_file_to_kb(file_record.id, kb_id)
        if success:
            logger.info(f"成功将新文档绑定到知识库: {filename}")
            return True, "新文档上传并绑定成功"
        else:
            logger.error(f"新文档上传成功但绑定失败: {filename} - {message}")
            return False, f"上传成功但绑定失败: {message}"

    except Exception as e:
        logger.error(f"上传文档到RAGFlow文件服务失败 {doc_name}: {e}")
        return False, str(e)


def get_clickup_docs():
    """获取ClickUp文档并根据创建时间过滤，上传到RAGFlow"""
    logger.info("开始同步ClickUp文档...")

    # 检查配置是否完整
    if not all([RAGFLOW_WEB_PARENT_FOLDER_ID, RAGFLOW_PRODUCT_PARENT_FOLDER_ID,
                RAGFLOW_WEB_KB_ID, RAGFLOW_PRODUCT_KB_ID]):
        logger.error("RAGFlow配置不完整，请先设置所有必要的ID")
        logger.error(f"RAGFLOW_WEB_PARENT_FOLDER_ID = {RAGFLOW_WEB_PARENT_FOLDER_ID}")
        logger.error(f"RAGFLOW_PRODUCT_PARENT_FOLDER_ID = {RAGFLOW_PRODUCT_PARENT_FOLDER_ID}")
        logger.error(f"RAGFLOW_WEB_KB_ID = {RAGFLOW_WEB_KB_ID}")
        logger.error(f"RAGFLOW_PRODUCT_KB_ID = {RAGFLOW_PRODUCT_KB_ID}")
        return 0, 1

    # 获取最后同步时间
    last_sync_time = get_last_sync_time()
    current_time = int(datetime.now().timestamp() * 1000)

    logger.info(f"最后同步时间: {datetime.fromtimestamp(last_sync_time / 1000)}")

    success_count = 0
    error_count = 0

    # 处理文件夹配置
    folders = [
        (COINEX_WEB_FOLDER_ID, RAGFLOW_WEB_PARENT_FOLDER_ID, RAGFLOW_WEB_KB_ID, "Web文档"),
        (COINEX_PRODUCT_FOLDER_ID, RAGFLOW_PRODUCT_PARENT_FOLDER_ID, RAGFLOW_PRODUCT_KB_ID, "产品文档")
    ]

    for clickup_folder_id, ragflow_parent_id, ragflow_kb_id, folder_desc in folders:
        logger.info(f"开始处理{folder_desc}...")

        # 获取所有页面（不过滤时间）
        pages_response = get_all_pages(clickup_folder_id)
        if not pages_response or 'docs' not in pages_response:
            logger.warning(f"没有找到{folder_desc}的页面")
            continue

        pages = pages_response['docs']
        logger.info(f"获取到 {len(pages)} 个{folder_desc}页面，开始时间过滤...")

        synced_docs_count = 0

        for page in pages:
            try:
                doc_name = page.get('name', 'unknown')
                doc_id = page.get('id')
                page_created = page.get('date_created')

                if not doc_id:
                    continue

                logger.debug(f"检查文档: {doc_name}, 创建时间: {page_created}")

                # 获取文档内容
                doc_content_response = get_doc_content(doc_id)
                if not doc_content_response:
                    error_count += 1
                    continue

                # 处理文档内容并根据时间过滤
                documents = process_doc_content(
                    doc_content_response,
                    doc_name,
                    last_sync_time
                )

                if not documents:
                    logger.debug(f"文档 {doc_name} 没有需要同步的内容")
                    continue

                synced_docs_count += len(documents)

                for doc in documents:
                    # 上传到文件服务并绑定到知识库
                    success, message = upload_doc_to_ragflow(
                        doc['content'],
                        doc['name'],
                        ragflow_parent_id,
                        ragflow_kb_id
                    )

                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.error(f"上传失败: {doc['name']} - {message}")

            except Exception as e:
                error_count += 1
                logger.error(f"处理文档失败 {doc_name}: {e}")

        logger.info(f"{folder_desc}处理完成，共同步 {synced_docs_count} 个文档")

    save_last_sync_time(current_time)

    logger.info(f"同步完成！成功: {success_count}, 失败: {error_count}")
    return success_count, error_count


def start_index():
    """启动文档索引，查看进度"""
    try:
        # 检查文件夹ID是否设置
        if not RAGFLOW_WEB_PARENT_FOLDER_ID or not RAGFLOW_PRODUCT_PARENT_FOLDER_ID:
            logger.error("RAGFlow文件夹ID未设置，无法启动索引")
            return False

        folder_configs = [
            (RAGFLOW_WEB_PARENT_FOLDER_ID, "Web文档"),
            (RAGFLOW_PRODUCT_PARENT_FOLDER_ID, "产品文档")
        ]

        total_processed = 0
        total_started = 0

        for folder_id, folder_name in folder_configs:
            logger.info(f"检查文件夹 {folder_name} (ID: {folder_id}) 中的文档")

            # 获取文件夹中所有来自ClickUp的文件（不限制文件类型）
            files = FileService.query(parent_id=folder_id, source_type="clickup")
            logger.info(f"文件夹 {folder_name} 中有 {len(files)} 个ClickUp文件")

            for file in files:
                total_processed += 1
                try:
                    # 检查文件是否已经绑定到文档
                    file_to_docs = File2DocumentService.get_by_file_id(file.id)

                    if not file_to_docs:
                        logger.debug(f"文件 {file.name} 未绑定到任何文档，跳过")
                        continue

                    # 获取关联的文档
                    for f2d in file_to_docs:
                        doc_id = f2d.document_id
                        e, doc = DocumentService.get_by_id(doc_id)

                        if not e:
                            logger.warning(f"找不到文档 {doc_id}，跳过")
                            continue

                        # 检查文档状态
                        if doc.run == TaskStatus.RUNNING.value:
                            logger.debug(f"文档 {doc.name} 已在处理中，跳过")
                            continue

                        if doc.progress >= 1.0:
                            logger.debug(f"文档 {doc.name} 已完成处理 (进度: {doc.progress})")
                            continue

                        logger.info(f"启动文档处理: {doc.name}")

                        # 设置文档为运行状态
                        update_info = {
                            "run": TaskStatus.RUNNING.value,
                            "progress": 0,
                            "progress_msg": "开始处理"
                        }

                        success = DocumentService.update_by_id(doc_id, update_info)
                        if not success:
                            logger.error(f"更新文档状态失败: {doc.name}")
                            continue

                        # 获取租户ID
                        tenant_id = DocumentService.get_tenant_id(doc_id)
                        if not tenant_id:
                            logger.error(f"获取租户ID失败: {doc.name}")
                            continue

                        # 准备文档信息用于队列处理
                        doc_dict = doc.to_dict()
                        doc_dict["tenant_id"] = tenant_id

                        # 获取文件存储地址
                        bucket, name = File2DocumentService.get_storage_address(doc_id=doc_id)

                        # 将文档加入处理队列
                        queue_tasks(doc_dict, bucket, name, 0)

                        total_started += 1
                        logger.info(f"文档 {doc.name} 已加入处理队列")

                except Exception as e:
                    logger.error(f"处理文件 {file.name} 时出错: {e}")
                    continue

        logger.info(f"索引启动完成！检查文件总数: {total_processed}, 启动处理文档数: {total_started}")

        return True

    except Exception as e:
        logger.error(f"启动索引失败: {e}")
        return False


def check_processing_status():
    """检查文档处理状态"""
    try:
        # 检查文件夹ID是否设置
        if not RAGFLOW_WEB_PARENT_FOLDER_ID or not RAGFLOW_PRODUCT_PARENT_FOLDER_ID:
            logger.warning("RAGFlow文件夹ID未设置")
            return

        folder_configs = [
            (RAGFLOW_WEB_PARENT_FOLDER_ID, "Web文档"),
            (RAGFLOW_PRODUCT_PARENT_FOLDER_ID, "产品文档")
        ]

        total_files = 0
        total_processing = 0
        total_completed = 0
        total_failed = 0

        for folder_id, folder_name in folder_configs:
            # 获取ClickUp文件（不限制文件类型）
            files = FileService.query(parent_id=folder_id, source_type="clickup")
            total_files += len(files)
            processing_count = 0
            completed_count = 0
            failed_count = 0

            for file in files:
                file_to_docs = File2DocumentService.get_by_file_id(file.id)
                for f2d in file_to_docs:
                    e, doc = DocumentService.get_by_id(f2d.document_id)
                    if e:
                        if doc.run == TaskStatus.RUNNING.value:
                            processing_count += 1
                        elif doc.progress >= 1.0:
                            completed_count += 1
                        elif doc.run == TaskStatus.FAIL.value:
                            failed_count += 1

            total_processing += processing_count
            total_completed += completed_count
            total_failed += failed_count

            logger.info(
                f"{folder_name}: {len(files)} 文件, {processing_count} 处理中, {completed_count} 已完成, {failed_count} 失败")

        logger.info(
            f"总计: {total_files} 文件, {total_processing} 处理中, {total_completed} 已完成, {total_failed} 失败")

    except Exception as e:
        logger.error(f"检查状态失败: {e}")


def check_sync_status():
    """检查同步状态和处理状态"""
    try:
        # 检查文件夹ID是否设置
        if not RAGFLOW_WEB_PARENT_FOLDER_ID or not RAGFLOW_PRODUCT_PARENT_FOLDER_ID:
            logger.warning("RAGFlow文件夹ID未设置")
            return

        folder_configs = [
            (RAGFLOW_WEB_PARENT_FOLDER_ID, "Web文档"),
            (RAGFLOW_PRODUCT_PARENT_FOLDER_ID, "产品文档")
        ]

        for folder_id, folder_name in folder_configs:
            files = FileService.query(parent_id=folder_id, source_type="clickup")
            logger.info(f"{folder_name}文件夹: {len(files)} 个ClickUp同步的文档")

        # 同时检查处理状态
        check_processing_status()

    except Exception as e:
        logger.error(f"检查状态失败: {e}")


def run():
    """通过schedule管理定时任务"""
    logger.info("启动ClickUp文档同步定时任务...")

    # 检查配置
    if not all([RAGFLOW_WEB_PARENT_FOLDER_ID, RAGFLOW_PRODUCT_PARENT_FOLDER_ID,
                RAGFLOW_WEB_KB_ID, RAGFLOW_PRODUCT_KB_ID]):
        logger.error("请先设置所有RAGFlow配置：")
        logger.error("RAGFLOW_WEB_PARENT_FOLDER_ID =")
        logger.error("RAGFLOW_PRODUCT_PARENT_FOLDER_ID =")
        logger.error("RAGFLOW_WEB_KB_ID =")
        logger.error("RAGFLOW_PRODUCT_KB_ID =")
        return

    # 立即执行一次同步
    get_clickup_docs()
    start_index()

    # 设置定时任务
    # 每天凌晨1点检查新文档
    schedule.every().day.at("01:00").do(get_clickup_docs)

    # 每天凌晨2点启动索引检查
    schedule.every().day.at("02:00").do(start_index)

    # 每30分钟检查一次状态
    schedule.every(30).minutes.do(check_sync_status)

    logger.info("定时任务已设置:")
    logger.info("- 每天凌晨1点同步ClickUp文档")
    logger.info("- 每天凌晨2点启动文档索引")
    logger.info("- 每30分钟检查一次同步状态")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        logger.info("停止同步任务")


if __name__ == "__main__":
    run()
