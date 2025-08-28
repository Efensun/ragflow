#!/usr/bin/env python3
"""
删除知识库中重复文档的脚本

用法:
python remove_duplicate_docs.py <kb_id>

删除规则:
1. 根据文档名称下划线分割后的最后一部分判断重复
2. 优先保留最后一行包含 yuque.com 的文档
3. 如果都不包含 yuque.com，优先保留创建时间最新的文档
"""

import logging
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api.db.db_models import DB, Document, Task
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.task_service import TaskService
from api.db import FileSource
from api.db.db_models import File
from rag.nlp.search import fetch_full_doc_from_storage
from rag.utils.storage_factory import STORAGE_IMPL

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_document_key(doc_name: str) -> str:
    """
    从文档名称中提取用于去重的关键部分
    例如: "业务文档_质押借币增加借币订单.md" -> "质押借币增加借币订单.md"
    """
    parts = doc_name.split('_')
    if len(parts) > 1:
        return parts[-1]
    return doc_name


def get_last_line_url(content: str) -> Optional[str]:
    """
    获取文档内容最后一行的URL
    """
    if not content:
        return None

    lines = content.strip().split('\n')
    if not lines:
        return None

    last_line = lines[-1].strip()

    # 查找各种URL模式
    url_patterns = [
        r'https?://[^\s<>"\)]+',  # 普通URL
        r'>\s*原文:\s*<([^>]+)>',  # > 原文: <URL>
        r'\[Origin URL\]\(([^)]+)\)',  # [Origin URL](URL)
        r'\[.*?\]\(([^)]+)\)',  # [任意文本](URL)
    ]

    for pattern in url_patterns:
        match = re.search(pattern, last_line)
        if match:
            if match.groups():
                return match.group(1)
            else:
                return match.group(0)

    return None


def choose_document_to_keep(docs: List[Dict]) -> Dict:
    """
    从重复文档列表中选择要保留的文档

    规则:
    1. 优先保留最后一行包含 yuque.com 的文档
    2. 如果都不包含或都包含 yuque.com，优先保留创建时间最新的文档
    """
    if len(docs) == 1:
        return docs[0]

    # 获取每个文档的内容和URL
    docs_with_info = []
    for doc in docs:
        content = fetch_full_doc_from_storage(doc['id'])
        last_url = get_last_line_url(content) if content else None
        has_yuque = last_url and 'yuque.com' in last_url if last_url else False

        docs_with_info.append({
            'doc': doc,
            'content': content,
            'last_url': last_url,
            'has_yuque': has_yuque,
        })

        # 格式化时间显示
        create_time = doc.get('create_time', 0)
        create_date = doc.get('create_date', 'N/A')

        logger.info(
            f"文档 {doc['name']}: yuque={has_yuque}, 创建时间={create_time}, 创建日期={create_date}, URL={last_url}")

    # 先按是否包含yuque.com分组
    yuque_docs = [d for d in docs_with_info if d['has_yuque']]
    non_yuque_docs = [d for d in docs_with_info if not d['has_yuque']]

    # 如果有包含yuque.com的文档，优先从这些文档中选择
    if yuque_docs:
        candidate_docs = yuque_docs
        logger.info(f"找到 {len(yuque_docs)} 个包含 yuque.com 的文档，从中选择")
    else:
        candidate_docs = non_yuque_docs
        logger.info(f"没有包含 yuque.com 的文档，从所有 {len(non_yuque_docs)} 个文档中选择")

    # 在候选文档中选择创建时间最新的（create_time是时间戳，越大越新）
    chosen = max(candidate_docs, key=lambda x: x['doc'].get('create_time', 0))

    create_time = chosen['doc'].get('create_time', 0)
    create_date = chosen['doc'].get('create_date', 'N/A')
    logger.info(f"选择文档: {chosen['doc']['name']} (创建时间: {create_time}, 创建日期: {create_date})")

    return chosen['doc']


def delete_document(doc_id: str, doc_name: str) -> bool:
    """
    删除指定的文档
    """
    try:
        logger.info(f"开始删除文档: {doc_name} (ID: {doc_id})")

        # 获取文档信息
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            logger.error(f"文档不存在: {doc_id}")
            return False

        # 获取租户ID
        tenant_id = DocumentService.get_tenant_id(doc_id)
        if not tenant_id:
            logger.error(f"无法获取租户ID: {doc_id}")
            return False

        # 获取存储地址
        bucket, name = File2DocumentService.get_storage_address(doc_id=doc_id)

        # 删除相关任务
        TaskService.filter_delete([Task.doc_id == doc_id])
        logger.info(f"已删除相关任务: {doc_name}")

        # 删除文档记录（这会同时清理索引）
        if not DocumentService.remove_document(doc, tenant_id):
            logger.error(f"删除文档记录失败: {doc_name}")
            return False

        # 删除文件到文档的绑定关系
        f2d = File2DocumentService.get_by_document_id(doc_id)
        if f2d:
            deleted_file_count = FileService.filter_delete([
                File.source_type == FileSource.KNOWLEDGEBASE,
                File.id == f2d[0].file_id
            ])
            File2DocumentService.delete_by_document_id(doc_id)

            # 删除存储文件
            if deleted_file_count > 0:
                try:
                    STORAGE_IMPL.rm(bucket, name)
                    logger.info(f"已删除存储文件: {bucket}/{name}")
                except Exception as e:
                    logger.warning(f"删除存储文件失败: {e}")
        else:
            File2DocumentService.delete_by_document_id(doc_id)

        logger.info(f"成功删除文档: {doc_name}")
        return True

    except Exception as e:
        logger.error(f"删除文档失败 {doc_name}: {e}")
        return False


def remove_duplicate_documents(kb_id: str) -> Tuple[int, int]:
    """
    删除知识库中的重复文档

    返回: (删除的文档数量, 处理的重复组数量)
    """
    logger.info(f"开始处理知识库 {kb_id} 的重复文档")

    # 获取知识库中的所有文档
    try:
        with DB.connection_context():
            docs = list(Document.select().where(Document.kb_id == kb_id).dicts())

        logger.info(f"知识库 {kb_id} 共有 {len(docs)} 个文档")

        if len(docs) == 0:
            logger.info("知识库为空，无需处理")
            return 0, 0

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return 0, 0

    # 按文档名称的关键部分分组
    grouped_docs = defaultdict(list)
    for doc in docs:
        key = get_document_key(doc['name'])
        grouped_docs[key].append(doc)

    # 找出重复的文档组
    duplicate_groups = {k: v for k, v in grouped_docs.items() if len(v) > 1}

    logger.info(f"发现 {len(duplicate_groups)} 组重复文档")

    if len(duplicate_groups) == 0:
        logger.info("没有发现重复文档")
        return 0, 0

    deleted_count = 0
    processed_groups = 0

    # 处理每组重复文档
    for key, doc_group in duplicate_groups.items():
        logger.info(f"\n处理重复组 '{key}': {len(doc_group)} 个文档")

        for doc in doc_group:
            create_time = doc.get('create_time', 0)
            create_date = doc.get('create_date', 'N/A')
            logger.info(f"  - {doc['name']} (ID: {doc['id']}, 创建时间: {create_time}, 创建日期: {create_date})")

        try:
            # 选择要保留的文档
            keep_doc = choose_document_to_keep(doc_group)
            logger.info(f"决定保留文档: {keep_doc['name']}")

            # 删除其他文档
            for doc in doc_group:
                if doc['id'] != keep_doc['id']:
                    if delete_document(doc['id'], doc['name']):
                        deleted_count += 1
                    else:
                        logger.error(f"删除文档失败: {doc['name']}")

            processed_groups += 1

        except Exception as e:
            logger.error(f"处理重复组 '{key}' 时出错: {e}")

    logger.info(f"\n处理完成: 删除了 {deleted_count} 个文档，处理了 {processed_groups} 组重复文档")
    return deleted_count, processed_groups


def main():
    if len(sys.argv) != 2:
        print("用法: python remove_duplicate_docs.py <kb_id>")
        print("例如: python remove_duplicate_docs.py your_knowledge_base_id")
        sys.exit(1)

    kb_id = sys.argv[1]

    logger.info(f"开始删除知识库 {kb_id} 的重复文档")

    try:
        deleted_count, processed_groups = remove_duplicate_documents(kb_id)

        if processed_groups > 0:
            logger.info(f"任务完成: 成功删除 {deleted_count} 个重复文档，处理了 {processed_groups} 组重复文档")
        else:
            logger.info("没有发现重复文档，无需删除")

    except Exception as e:
        logger.error(f"执行过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()