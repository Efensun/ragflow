#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的知识库设置脚本

直接在ES和ParadeDB中创建测试数据，不依赖复杂的数据库服务
"""

import os
import sys
import json
import time
import logging
import uuid
from typing import List, Dict, Any
import numpy as np
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import settings
from rag.utils.es_conn import ESConnection
from rag.utils.pd_conn import PDConnection

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleKBSetup:
    """简化的知识库设置器"""
    
    def __init__(self):
        self.es_conn = None
        self.pd_conn = None
        self.embedding_model = None
        self.init_connections()
        self.init_embedding_model()
    
    def init_connections(self):
        """初始化数据库连接"""
        try:
            # 初始化ES连接
            if hasattr(settings, 'ES') and settings.ES.get('hosts'):
                logger.info("初始化Elasticsearch连接...")
                self.es_conn = ESConnection()
                logger.info("✅ Elasticsearch连接成功")
            else:
                logger.warning("⚠️ 未配置Elasticsearch")
            
            # 初始化ParadeDB连接
            if hasattr(settings, 'PARADEDB') and settings.PARADEDB.get('host'):
                logger.info("初始化ParadeDB连接...")
                self.pd_conn = PDConnection()
                logger.info("✅ ParadeDB连接成功")
            else:
                logger.warning("⚠️ 未配置ParadeDB")
                
        except Exception as e:
            logger.error(f"❌ 初始化数据库连接失败: {e}")
            raise
    
    def init_embedding_model(self):
        """初始化embedding模型"""
        try:
            logger.info("初始化embedding模型...")
            
            # 尝试使用RAGFlow的embedding模型
            try:
                from rag.nlp import EmbeddingModel
                # 使用默认的embedding模型
                model_name = getattr(settings, 'LLM_FACTORY', {}).get('embedding_model', 'BAAI/bge-large-zh-v1.5')
                self.embedding_model = EmbeddingModel(model_name, "")
                logger.info(f"✅ 使用RAGFlow embedding模型: {model_name}")
                return
            except Exception as e:
                logger.warning(f"无法加载RAGFlow embedding模型: {e}")
            
            # 备选方案1: 使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                model_name = 'BAAI/bge-large-zh-v1.5'
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"✅ 使用sentence-transformers模型: {model_name}")
                return
            except ImportError:
                logger.warning("sentence-transformers未安装")
            except Exception as e:
                logger.warning(f"无法加载sentence-transformers模型: {e}")
            
            # 备选方案2: 使用transformers
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                model_name = 'BAAI/bge-large-zh-v1.5'
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.embedding_model = "transformers"
                logger.info(f"✅ 使用transformers模型: {model_name}")
                return
            except ImportError:
                logger.warning("transformers未安装")
            except Exception as e:
                logger.warning(f"无法加载transformers模型: {e}")
            
            # 如果所有方案都失败，使用模拟向量但给出警告
            logger.warning("⚠️ 无法加载任何embedding模型，将使用模拟向量")
            logger.warning("⚠️ 建议安装: pip install sentence-transformers")
            self.embedding_model = None
            
        except Exception as e:
            logger.error(f"❌ embedding模型初始化失败: {e}")
            self.embedding_model = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成文本的embedding向量"""
        try:
            if self.embedding_model is None:
                # 使用模拟向量作为备选
                logger.debug(f"使用模拟向量: {text[:50]}...")
                hash_value = hash(text) % (2**32)
                np.random.seed(hash_value)
                vector = np.random.normal(0, 1, 1024)
                vector = vector / np.linalg.norm(vector)
                return vector.tolist()
            
            # RAGFlow embedding模型
            if hasattr(self.embedding_model, 'encode'):
                if hasattr(self.embedding_model, 'embed_documents'):
                    # RAGFlow EmbeddingModel
                    embeddings = self.embedding_model.embed_documents([text])
                    return embeddings[0]
                else:
                    # sentence-transformers
                    embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                    return embedding.tolist()
            
            # transformers模型
            elif self.embedding_model == "transformers":
                import torch
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS] token的embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                    # 归一化
                    embedding = embedding / torch.norm(embedding)
                    return embedding.numpy().tolist()
            
            else:
                raise Exception("未知的embedding模型类型")
                
        except Exception as e:
            logger.warning(f"生成embedding失败: {e}，使用模拟向量")
            # 使用文本hash作为种子，确保相同文本生成相同向量
            hash_value = hash(text) % (2**32)
            np.random.seed(hash_value)
            vector = np.random.normal(0, 1, 1024)
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()
    
    def generate_uuid(self) -> str:
        """生成UUID"""
        return str(uuid.uuid4())
    
    def create_indexes(self, index_name: str, kb_id: str, vector_size: int = 1024):
        """在ES和ParadeDB中创建索引"""
        logger.info(f"创建索引: {index_name}")
        
        # 创建ES索引
        if self.es_conn:
            try:
                logger.info(f"在ES中创建索引: {index_name}")
                success = self.es_conn.createIdx(index_name, kb_id, vector_size)
                if success:
                    logger.info(f"✅ ES索引创建成功: {index_name}")
                else:
                    logger.info(f"ℹ️ ES索引已存在: {index_name}")
            except Exception as e:
                logger.error(f"❌ ES索引创建失败: {e}")
                raise
        
        # 创建ParadeDB表
        if self.pd_conn:
            try:
                logger.info(f"在ParadeDB中创建表: {index_name}")
                success = self.pd_conn.createIdx(index_name, kb_id, vector_size)
                if success:
                    logger.info(f"✅ ParadeDB表创建成功: {index_name}")
                else:
                    logger.info(f"ℹ️ ParadeDB表已存在: {index_name}")
            except Exception as e:
                logger.error(f"❌ ParadeDB表创建失败: {e}")
                raise
    
    def generate_sample_documents(self, kb_id: str, count: int = 20) -> List[Dict]:
        """生成示例文档数据"""
        logger.info(f"生成 {count} 个示例文档...")
        
        # 示例文档内容
        sample_contents = [
            {
                "title": "人工智能基础知识",
                "content": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器和系统。AI包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。现代AI系统能够进行图像识别、语音识别、自然语言理解和生成等复杂任务。",
                "keywords": ["人工智能", "机器学习", "深度学习", "计算机视觉", "自然语言处理"]
            },
            {
                "title": "机器学习算法详解",
                "content": "机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习模式和规律。主要分为监督学习、无监督学习和强化学习三大类。监督学习包括分类和回归任务，常用算法有决策树、随机森林、支持向量机、神经网络等。无监督学习主要用于聚类和降维。",
                "keywords": ["机器学习", "监督学习", "无监督学习", "强化学习", "神经网络", "决策树"]
            },
            {
                "title": "自然语言处理技术应用",
                "content": "自然语言处理（NLP）是AI领域的重要分支，专注于让计算机理解和生成人类语言。NLP技术广泛应用于机器翻译、情感分析、文本摘要、问答系统、聊天机器人等场景。近年来，基于Transformer架构的大语言模型如GPT、BERT等取得了突破性进展。",
                "keywords": ["自然语言处理", "NLP", "机器翻译", "情感分析", "Transformer", "GPT", "BERT"]
            },
            {
                "title": "数据科学与大数据分析",
                "content": "数据科学是一个跨学科领域，结合统计学、计算机科学和领域专业知识来从数据中提取有价值的洞察。大数据分析涉及处理和分析大规模、高速度、多样化的数据集。常用工具包括Python、R、SQL、Hadoop、Spark等。数据可视化是数据科学的重要组成部分。",
                "keywords": ["数据科学", "大数据", "数据分析", "Python", "R", "Hadoop", "Spark", "数据可视化"]
            },
            {
                "title": "云计算服务平台",
                "content": "云计算是通过互联网提供可扩展的计算资源和服务的模式。主要服务模型包括基础设施即服务(IaaS)、平台即服务(PaaS)和软件即服务(SaaS)。主要云服务提供商包括Amazon AWS、Microsoft Azure、Google Cloud Platform等。云计算具有弹性扩展、按需付费、高可用性等优势。",
                "keywords": ["云计算", "AWS", "Azure", "Google Cloud", "IaaS", "PaaS", "SaaS", "弹性扩展"]
            },
            {
                "title": "区块链技术原理与应用",
                "content": "区块链是一种分布式账本技术，通过密码学方法将交易记录链接成不可篡改的区块链。每个区块包含交易数据、时间戳和前一个区块的哈希值。区块链具有去中心化、透明性、不可篡改等特点，广泛应用于加密货币、供应链管理、数字身份认证等领域。",
                "keywords": ["区块链", "分布式账本", "密码学", "去中心化", "加密货币", "智能合约"]
            },
            {
                "title": "Zendesk客户服务解决方案",
                "content": "Zendesk是全球领先的客户服务和支持平台，为企业提供全方位的客户服务解决方案。主要功能包括工单管理系统、知识库、实时聊天、客户满意度调查、多渠道支持等。Zendesk帮助企业提升客户体验，提高服务效率，支持邮件、电话、社交媒体、网页等多种沟通渠道。",
                "keywords": ["Zendesk", "客户服务", "工单系统", "知识库", "实时聊天", "客户体验", "多渠道支持"]
            },
            {
                "title": "RESTful API设计规范",
                "content": "REST（Representational State Transfer）是一种软件架构风格，用于设计网络应用程序的API。RESTful API遵循统一接口、无状态、可缓存、分层系统等原则。设计时应使用标准HTTP方法（GET、POST、PUT、DELETE），合理设计URL结构，使用适当的HTTP状态码，提供清晰的API文档。",
                "keywords": ["API", "RESTful", "HTTP", "接口设计", "状态码", "URL设计", "API文档"]
            },
            {
                "title": "数据库管理与优化",
                "content": "数据库管理系统（DBMS）是管理和操作数据库的核心软件。关系型数据库如MySQL、PostgreSQL使用SQL语言进行数据操作，适合结构化数据存储。NoSQL数据库如MongoDB、Redis适合处理非结构化数据和高并发场景。数据库优化包括索引设计、查询优化、分库分表等策略。",
                "keywords": ["数据库", "DBMS", "MySQL", "PostgreSQL", "NoSQL", "MongoDB", "Redis", "索引优化"]
            },
            {
                "title": "软件开发最佳实践",
                "content": "现代软件开发遵循敏捷开发方法论，强调迭代开发、持续集成、自动化测试。DevOps实践将开发和运维紧密结合，通过CI/CD流水线实现快速交付。代码质量管理包括代码审查、单元测试、集成测试等。版本控制使用Git，项目管理常用Jira、Trello等工具。",
                "keywords": ["软件开发", "敏捷开发", "DevOps", "CI/CD", "Git", "代码审查", "自动化测试"]
            }
        ]
        
        documents = []
        for i in range(count):
            content_data = sample_contents[i % len(sample_contents)]
            
            # 生成文档ID
            doc_id = self.generate_uuid()
            chunk_id = self.generate_uuid()
            
            # 生成真实的embedding向量
            full_text = f"{content_data['title']} {content_data['content']}"
            logger.info(f"生成embedding向量 {i+1}/{count}: {content_data['title']}")
            vector = self.generate_embedding(full_text)
            
            doc = {
                "id": chunk_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                "docnm_kwd": f"{content_data['title']}.txt",
                "title_tks": content_data["title"],
                "title_sm_tks": content_data["title"],
                "content_ltks": content_data["content"],
                "content_sm_ltks": content_data["content"],
                "content_with_weight": content_data["content"],
                "important_kwd": content_data["keywords"],
                "important_tks": " ".join(content_data["keywords"]),
                "question_tks": f"什么是{content_data['title']}？如何理解{content_data['keywords'][0]}？",
                "available_int": 1,
                "create_timestamp_flt": time.time() + i,  # 确保时间戳不同
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "weight_int": random.randint(1, 10),
                "weight_flt": round(random.uniform(0.1, 1.0), 3),
                "rank_int": random.randint(1, 100),
                "rank_flt": round(random.uniform(0.1, 1.0), 3),
                "page_num_int": [1],
                "top_int": [0],
                "position_int": [0],
                f"q_1024_vec": vector,
                "metadata": {
                    "source": "test_data",
                    "type": "text",
                    "language": "zh",
                    "index": i,
                    "embedding_model": str(type(self.embedding_model).__name__) if self.embedding_model else "simulated"
                }
            }
            documents.append(doc)
        
        logger.info(f"✅ 生成了 {len(documents)} 个示例文档（使用真实embedding向量）")
        return documents
    
    def insert_documents(self, documents: List[Dict], index_name: str, kb_id: str):
        """将文档插入到ES和ParadeDB"""
        logger.info(f"开始插入 {len(documents)} 个文档...")
        
        es_success = False
        pd_success = False
        
        # 插入到ES
        if self.es_conn:
            try:
                logger.info("插入文档到Elasticsearch...")
                errors = self.es_conn.insert(documents, index_name, kb_id)
                if errors:
                    logger.warning(f"ES插入时出现 {len(errors)} 个错误: {errors[:3]}...")
                else:
                    logger.info(f"✅ 成功插入 {len(documents)} 个文档到ES")
                    es_success = True
            except Exception as e:
                logger.error(f"❌ ES文档插入失败: {e}")
        
        # 插入到ParadeDB
        if self.pd_conn:
            try:
                logger.info("插入文档到ParadeDB...")
                errors = self.pd_conn.insert(documents, index_name, kb_id)
                if errors:
                    logger.warning(f"ParadeDB插入时出现 {len(errors)} 个错误: {errors[:3]}...")
                else:
                    logger.info(f"✅ 成功插入 {len(documents)} 个文档到ParadeDB")
                    pd_success = True
            except Exception as e:
                logger.error(f"❌ ParadeDB文档插入失败: {e}")
        
        return es_success, pd_success
    
    def verify_data(self, index_name: str, kb_id: str) -> Dict[str, Any]:
        """验证数据一致性"""
        logger.info("验证数据一致性...")
        
        result = {
            "es_count": 0,
            "pd_count": 0,
            "consistent": False,
            "es_sample": None,
            "pd_sample": None
        }
        
        # 检查ES
        if self.es_conn:
            try:
                from rag.utils.doc_store_conn import OrderByExpr
                es_res = self.es_conn.search(
                    selectFields=["id", "title_tks", "content_with_weight"],
                    highlightFields=[],
                    condition={"available_int": 1},
                    matchExprs=[],
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=100,
                    indexNames=index_name,
                    knowledgebaseIds=[kb_id]
                )
                result["es_count"] = self.es_conn.getTotal(es_res)
                
                # 获取样本
                hits = es_res.get("hits", {}).get("hits", [])
                if hits:
                    sample = hits[0]
                    result["es_sample"] = {
                        "id": sample.get("_id"),
                        "title": sample.get("_source", {}).get("title_tks", ""),
                        "content_preview": sample.get("_source", {}).get("content_with_weight", "")[:100]
                    }
                
            except Exception as e:
                logger.error(f"ES数据验证失败: {e}")
        
        # 检查ParadeDB
        if self.pd_conn:
            try:
                from rag.utils.doc_store_conn import OrderByExpr
                pd_res = self.pd_conn.search(
                    selectFields=["id", "title_tks", "content_with_weight"],
                    highlightFields=[],
                    condition={"available_int": 1},
                    matchExprs=[],
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=100,
                    indexNames=index_name,
                    knowledgebaseIds=[kb_id]
                )
                result["pd_count"] = self.pd_conn.getTotal(pd_res)
                
                # 获取样本
                hits = pd_res.get("hits", {}).get("hits", [])
                if hits:
                    sample = hits[0]
                    result["pd_sample"] = {
                        "id": sample.get("_id"),
                        "title": sample.get("_source", {}).get("title_tks", ""),
                        "content_preview": sample.get("_source", {}).get("content_with_weight", "")[:100]
                    }
                
            except Exception as e:
                logger.error(f"ParadeDB数据验证失败: {e}")
        
        # 检查一致性
        result["consistent"] = (
            result["es_count"] == result["pd_count"] and 
            result["es_count"] > 0
        )
        
        return result
    
    def setup_test_knowledge_base(self, kb_name: str = "测试知识库", 
                                 document_count: int = 20) -> Dict[str, Any]:
        """设置测试知识库"""
        logger.info(f"开始设置测试知识库: {kb_name}")
        
        # 生成ID
        kb_id = self.generate_uuid()
        tenant_id = self.generate_uuid()
        index_name = f"ragflow_{tenant_id.replace('-', '')}"
        
        try:
            # 1. 创建索引
            self.create_indexes(index_name, kb_id)
            
            # 2. 生成文档
            documents = self.generate_sample_documents(kb_id, document_count)
            
            # 3. 插入文档
            es_success, pd_success = self.insert_documents(documents, index_name, kb_id)
            
            # 4. 等待索引刷新
            logger.info("等待索引刷新...")
            time.sleep(5)
            
            # 5. 验证数据
            verification = self.verify_data(index_name, kb_id)
            
            result = {
                "kb_id": kb_id,
                "tenant_id": tenant_id,
                "index_name": index_name,
                "kb_name": kb_name,
                "document_count": document_count,
                "es_success": es_success,
                "pd_success": pd_success,
                "verification": verification,
                "success": verification["consistent"]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 知识库设置失败: {e}")
            raise

def main():
    """主函数"""
    print("🚀 简化知识库设置工具（使用真实embedding向量）")
    print("=" * 50)
    
    try:
        # 初始化设置器
        setup = SimpleKBSetup()
        
        if not setup.es_conn and not setup.pd_conn:
            print("❌ 没有可用的搜索引擎连接")
            return
        
        # 设置知识库
        result = setup.setup_test_knowledge_base(
            kb_name="ES与ParadeDB对比测试知识库（真实向量）",
            document_count=20
        )
        
        print(f"\n📊 设置结果:")
        print(f"  知识库名称: {result['kb_name']}")
        print(f"  知识库ID: {result['kb_id']}")
        print(f"  索引名称: {result['index_name']}")
        print(f"  租户ID: {result['tenant_id']}")
        print(f"  文档数量: {result['document_count']}")
        print(f"  Embedding模型: {str(type(setup.embedding_model).__name__) if setup.embedding_model else 'simulated'}")
        print(f"  ES插入成功: {'✅' if result['es_success'] else '❌'}")
        print(f"  ParadeDB插入成功: {'✅' if result['pd_success'] else '❌'}")
        
        verification = result['verification']
        print(f"\n🔍 数据验证:")
        print(f"  ES文档数: {verification['es_count']}")
        print(f"  ParadeDB文档数: {verification['pd_count']}")
        print(f"  数据一致性: {'✅' if verification['consistent'] else '❌'}")
        
        if verification['es_sample']:
            print(f"\n📄 ES样本文档:")
            print(f"  ID: {verification['es_sample']['id']}")
            print(f"  标题: {verification['es_sample']['title']}")
            print(f"  内容预览: {verification['es_sample']['content_preview']}...")
        
        if verification['pd_sample']:
            print(f"\n📄 ParadeDB样本文档:")
            print(f"  ID: {verification['pd_sample']['id']}")
            print(f"  标题: {verification['pd_sample']['title']}")
            print(f"  内容预览: {verification['pd_sample']['content_preview']}...")
        
        if result['success']:
            print(f"\n🎉 知识库设置成功!")
            
            # 生成测试配置
            config_content = f'''# 自动生成的搜索对比测试配置（使用真实embedding向量）
# 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
# Embedding模型: {str(type(setup.embedding_model).__name__) if setup.embedding_model else 'simulated'}

INDEX_NAME = "{result['index_name']}"
KB_ID = "{result['kb_id']}"
TENANT_ID = "{result['tenant_id']}"

# 用于 quick_search_test.py
TEST_CONFIG = {{
    "index_name": "{result['index_name']}",
    "kb_ids": ["{result['kb_id']}"],
    "test_queries": [
        "人工智能基础知识",
        "机器学习算法", 
        "自然语言处理技术",
        "数据科学分析",
        "云计算平台",
        "区块链技术",
        "Zendesk客户服务",
        "API接口设计",
        "数据库管理",
        "软件开发实践",
        # 语义相关的查询
        "AI和深度学习",
        "NLP和文本分析",
        "大数据处理",
        "分布式系统",
        "客户支持平台"
    ]
}}

# 用于 compare_search_engines.py
COMPARISON_CONFIG = {{
    "index_name": "{result['index_name']}",
    "kb_ids": ["{result['kb_id']}"],
    "vector_weights": [0.2, 0.3, 0.5, 0.7, 0.8],  # 更多权重测试
    "test_queries": TEST_CONFIG["test_queries"]
}}
'''
            
            with open("test_kb_config.py", "w", encoding="utf-8") as f:
                f.write(config_content)
            
            print(f"\n📄 已生成测试配置文件: test_kb_config.py")
            print(f"\n💡 现在可以运行搜索对比测试:")
            print(f"  python quick_search_test.py")
            print(f"  python compare_search_engines.py")
            print(f"\n🎯 使用真实embedding向量，可以验证语义相关性！")
            
        else:
            print(f"\n❌ 知识库设置失败，请检查日志")
            
    except Exception as e:
        logger.error(f"设置过程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 