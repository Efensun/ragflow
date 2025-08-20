#!/usr/bin/env python3
"""
文档解析启动脚本
查找产品和Web文档中未开始解析的文档，批量启动解析任务
每次启动10个文档，间隔10分钟
"""

import logging
import os
import sys
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import TaskStatus
from api.db.db_models import Task
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.db.services.task_service import TaskService, queue_tasks
from api import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置信息 - 从现有脚本复制
RAGFLOW_WEB_KB_ID = '345887da660e11f09a61cf9d445194dc'  # Web文档知识库ID
RAGFLOW_PRODUCT_KB_ID = '8076260e660e11f09a61cf9d445194dc'  # 产品文档知识库ID

# 批处理配置
BATCH_SIZE = 10  # 每次启动10个文档
WAIT_INTERVAL = 600  # 10分钟 = 600秒


def init_settings():
    """初始化RAGFlow设置"""
    try:
        settings.init_settings()
        logger.info("RAGFlow 设置初始化成功")
        return True
    except Exception as e:
        logger.error(f"RAGFlow 设置初始化失败: {e}")
        return False


def get_unstarted_documents():
    """获取未开始解析的文档"""
    unstarted_docs = []

    knowledge_bases = [
        (RAGFLOW_WEB_KB_ID, "Web文档"),
        (RAGFLOW_PRODUCT_KB_ID, "产品文档")
    ]

    for kb_id, kb_name in knowledge_bases:
        try:
            # 查询状态为未开始的文档
            docs = DocumentService.query(
                kb_id=kb_id,
                run=TaskStatus.UNSTART.value,
                progress=0.0
            )

            logger.info(f"{kb_name}知识库中有 {len(docs)} 个未开始解析的文档")

            for doc in docs:
                unstarted_docs.append({
                    'doc': doc,
                    'kb_name': kb_name,
                    'kb_id': kb_id
                })

        except Exception as e:
            logger.error(f"查询{kb_name}知识库文档时出错: {e}")

    return unstarted_docs


def start_document_parsing(doc_info):
    """启动单个文档的解析"""
    doc = doc_info['doc']
    kb_name = doc_info['kb_name']

    try:
        logger.info(f"开始启动文档解析: {doc.name} (来自{kb_name})")

        # 清除旧的处理记录
        try:
            from rag.nlp import search

            tenant_id = DocumentService.get_tenant_id(doc.id)
            if tenant_id:
                # 删除现有任务
                TaskService.filter_delete([Task.doc_id == doc.id])

                # 清除索引数据
                if settings.docStoreConn.indexExist(search.index_name(tenant_id), doc.kb_id):
                    settings.docStoreConn.delete({"doc_id": doc.id}, search.index_name(tenant_id), doc.kb_id)

        except Exception as clean_e:
            logger.warning(f"清除旧数据失败: {clean_e}")

        # 设置文档为运行状态
        update_info = {
            "run": TaskStatus.RUNNING.value,
            "progress": 0,
            "progress_msg": "开始处理",
            "chunk_num": 0,
            "token_num": 0
        }

        success = DocumentService.update_by_id(doc.id, update_info)
        if not success:
            logger.error(f"更新文档状态失败: {doc.name}")
            return False

        # 获取租户ID
        tenant_id = DocumentService.get_tenant_id(doc.id)
        if not tenant_id:
            logger.error(f"获取租户ID失败: {doc.name}")
            return False

        # 准备文档信息用于队列处理
        doc_dict = doc.to_dict()
        doc_dict["tenant_id"] = tenant_id

        # 获取文件存储地址
        bucket, name = File2DocumentService.get_storage_address(doc_id=doc.id)

        # 将文档加入处理队列
        queue_tasks(doc_dict, bucket, name, 0)

        logger.info(f"文档 {doc.name} 已成功加入处理队列")
        return True

    except Exception as e:
        logger.error(f"启动文档 {doc.name} 解析时出错: {e}")
        # 如果出错，将文档状态重置为未开始
        try:
            DocumentService.update_by_id(doc.id, {
                "run": TaskStatus.UNSTART.value,
                "progress": 0,
                "progress_msg": f"启动失败: {str(e)}"
            })
        except:
            pass
        return False


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("文档解析启动脚本开始运行")
    logger.info("=" * 60)

    # 初始化设置
    if not init_settings():
        logger.error("设置初始化失败，程序退出")
        sys.exit(1)

    total_started = 0
    batch_count = 0

    while True:
        batch_count += 1
        logger.info(f"开始第 {batch_count} 批次处理...")

        # 获取未开始解析的文档
        unstarted_docs = get_unstarted_documents()

        if not unstarted_docs:
            logger.info("没有找到未开始解析的文档，程序结束")
            break

        logger.info(f"找到 {len(unstarted_docs)} 个未开始解析的文档")

        # 取前BATCH_SIZE个文档进行处理
        batch_docs = unstarted_docs[:BATCH_SIZE]
        batch_success = 0

        for doc_info in batch_docs:
            if start_document_parsing(doc_info):
                batch_success += 1
                total_started += 1

            # 每个文档之间稍微等待，避免系统压力过大
            time.sleep(1)

        logger.info(f"第 {batch_count} 批次完成，成功启动 {batch_success}/{len(batch_docs)} 个文档")
        logger.info(f"累计已启动 {total_started} 个文档")

        # 如果还有剩余文档需要处理，等待指定时间
        remaining_docs = len(unstarted_docs) - len(batch_docs)
        if remaining_docs > 0:
            logger.info(f"还有 {remaining_docs} 个文档等待处理，{WAIT_INTERVAL // 60} 分钟后继续...")
            time.sleep(WAIT_INTERVAL)
        else:
            logger.info("所有未开始解析的文档都已启动处理")
            break

    logger.info("=" * 60)
    logger.info(f"文档解析启动脚本完成，共启动了 {total_started} 个文档的解析")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("收到中断信号，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        sys.exit(1)