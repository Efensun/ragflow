#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速搜索引擎对比测试脚本（支持真实embedding向量）

这是一个简化版本，用于快速验证ES和ParadeDB的基本搜索功能，使用真实embedding进行语义搜索测试
"""

import os
import sys
import time
import logging
from typing import List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import settings
from rag.utils.es_conn import ESConnection
from rag.utils.pd_conn import PDConnection
from rag.utils.doc_store_conn import MatchTextExpr, MatchDenseExpr, FusionExpr, OrderByExpr
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickSearchTester:
    """快速搜索测试器"""
    
    def __init__(self):
        self.es_conn = None
        self.pd_conn = None
        self.embedding_model = None
        self.vector_dimension = 1024  # 默认向量维度
        self.init_connections()
        self.init_embedding_model()
    
    def init_connections(self):
        """初始化数据库连接"""
    try:
            # 初始化ES连接
        if hasattr(settings, 'ES') and settings.ES.get('hosts'):
            print("🔗 连接Elasticsearch...")
                self.es_conn = ESConnection()
            print("✅ ES连接成功")
        else:
            print("⚠️ 未配置ES，跳过ES测试")
    except Exception as e:
        print(f"❌ ES连接失败: {e}")
    
    try:
            # 初始化ParadeDB连接
        if hasattr(settings, 'PARADEDB') and settings.PARADEDB.get('host'):
            print("🔗 连接ParadeDB...")
                self.pd_conn = PDConnection()
            print("✅ ParadeDB连接成功")
        else:
            print("⚠️ 未配置ParadeDB，跳过ParadeDB测试")
    except Exception as e:
        print(f"❌ ParadeDB连接失败: {e}")
    
    def init_embedding_model(self):
        """初始化embedding模型（与simple_kb_setup.py保持一致）"""
        try:
            # 方法1: 直接使用OpenAI客户端调用Xinference（推荐，已验证可用）
            try:
                from openai import OpenAI
                
                # 使用用户提供的Xinference地址
                xinference_client = OpenAI(
                    api_key="empty", 
                    base_url="http://120.77.38.66:8008/v1"
                )
                
                # 测试模型是否可用
                test_response = xinference_client.embeddings.create(
                    input=["测试文本"],
                    model="jina-embeddings-v3"
                )
                
                if test_response.data and len(test_response.data[0].embedding) > 0:
                    self.embedding_model = xinference_client
                    self.vector_dimension = len(test_response.data[0].embedding)
                    print(f"✅ 使用OpenAI客户端直接调用Xinference jina-embeddings-v3模型，向量维度: {self.vector_dimension}")
                    return
                    
            except Exception as e:
                logger.warning(f"无法使用OpenAI客户端调用Xinference: {e}")
            
            # 方法2: 使用用户提供的租户ID和embedding配置（备选）
            try:
                from api.db.services.llm_service import LLMBundle, LLMType
                
                # 使用用户提供的租户ID
                tenant_id = "c68cf3243ba311f08ca03fb4f23258b9"
                print(f"尝试使用用户提供的租户ID: {tenant_id}")
                
                # 尝试创建embedding模型（使用配置的jina-embeddings-v3@Xinference）
                try:
                    embedding_bundle = LLMBundle(tenant_id, LLMType.EMBEDDING.value)
                    
                    # 测试模型是否可用
                    test_embeddings, _ = embedding_bundle.encode(["测试文本"])
                    if test_embeddings and len(test_embeddings[0]) > 0:
                        self.embedding_model = embedding_bundle
                        self.vector_dimension = len(test_embeddings[0])
                        print(f"✅ 使用RAGFlow配置的jina-embeddings-v3模型，向量维度: {self.vector_dimension}")
                        return
                except Exception as e:
                    logger.warning(f"无法使用RAGFlow LLMBundle: {e}")
                    
            except Exception as e:
                logger.warning(f"无法导入RAGFlow LLMBundle: {e}")
            
            # 方法3: 尝试使用RAGFlow的XinferenceEmbed类（备选）
            try:
                from rag.llm.embedding_model import XinferenceEmbed
                
                xinference_embed = XinferenceEmbed(
                    key="",  # Xinference通常不需要API key
                    model_name="jina-embeddings-v3",
                    base_url="http://120.77.38.66:8008/"
                )
                
                # 测试embedding
                test_embeddings, tokens = xinference_embed.encode(["测试文本"])
                if test_embeddings and len(test_embeddings[0]) > 0:
                    self.embedding_model = xinference_embed
                    self.vector_dimension = len(test_embeddings[0])
                    print(f"✅ 使用RAGFlow XinferenceEmbed模型，向量维度: {self.vector_dimension}")
                    return
                    
            except Exception as e:
                logger.warning(f"无法使用RAGFlow XinferenceEmbed: {e}")
            
            # 方法4: 尝试直接使用RAGFlow的embedding模型类
            try:
                from rag.llm import EmbeddingModel
                
                # 尝试使用jina-embeddings-v3模型
                if "Xinference" in EmbeddingModel:
                    try:
                        # 假设jina-embeddings-v3通过Xinference提供
                        xinference_embed = EmbeddingModel["Xinference"](
                            api_key="", 
                            model_name="jina-embeddings-v3",
                            base_url="http://120.77.38.66:8008/"
                        )
                        test_embeddings, _ = xinference_embed.encode(["测试文本"])
                        if test_embeddings and len(test_embeddings[0]) > 0:
                            self.embedding_model = xinference_embed
                            self.vector_dimension = len(test_embeddings[0])
                            print(f"✅ 使用Xinference jina-embeddings-v3模型，向量维度: {self.vector_dimension}")
                            return
                    except Exception as e:
                        logger.warning(f"无法使用Xinference jina-embeddings-v3: {e}")
                
                # 尝试使用BAAI模型
                if "BAAI" in EmbeddingModel:
                    try:
                        baai_embed = EmbeddingModel["BAAI"](
                            api_key="", 
                            model_name="BAAI/bge-large-zh-v1.5"
                        )
                        test_embeddings, _ = baai_embed.encode(["测试文本"])
                        if test_embeddings and len(test_embeddings[0]) > 0:
                            self.embedding_model = baai_embed
                            self.vector_dimension = len(test_embeddings[0])
                            print(f"✅ 使用BAAI embedding模型，向量维度: {self.vector_dimension}")
                            return
                    except Exception as e:
                        logger.warning(f"无法使用BAAI embedding模型: {e}")
                        
            except Exception as e:
                logger.warning(f"无法加载RAGFlow embedding模型: {e}")
            
            # 方法5: 备选方案 - 使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                
                # 优先尝试jina-embeddings-v3
                try:
                    self.embedding_model = SentenceTransformer('jinaai/jina-embeddings-v3')
                    test_embedding = self.embedding_model.encode("测试文本", normalize_embeddings=True)
                    self.vector_dimension = len(test_embedding)
                    print(f"✅ 使用sentence-transformers jina-embeddings-v3模型，向量维度: {self.vector_dimension}")
                    return
                except Exception as e:
                    logger.warning(f"无法加载jina-embeddings-v3: {e}")
                
                # 备选BAAI模型
                model_name = 'BAAI/bge-large-zh-v1.5'
                self.embedding_model = SentenceTransformer(model_name)
                test_embedding = self.embedding_model.encode("测试文本", normalize_embeddings=True)
                self.vector_dimension = len(test_embedding)
                print(f"✅ 使用sentence-transformers模型: {model_name}，向量维度: {self.vector_dimension}")
                return
            except ImportError:
                logger.warning("sentence-transformers未安装，建议安装: pip install sentence-transformers")
            except Exception as e:
                logger.warning(f"无法加载sentence-transformers模型: {e}")
            
            # 方法6: 备选方案 - 使用transformers
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                # 优先尝试jina-embeddings-v3
                try:
                    model_name = 'jinaai/jina-embeddings-v3'
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.embedding_model = "transformers"
                    
                    # 测试获取向量维度
                    inputs = self.tokenizer("测试文本", return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                        self.vector_dimension = len(embedding)
                    
                    print(f"✅ 使用transformers jina-embeddings-v3模型，向量维度: {self.vector_dimension}")
                    return
                except Exception as e:
                    logger.warning(f"无法加载transformers jina-embeddings-v3: {e}")
                
                # 备选BAAI模型
                model_name = 'BAAI/bge-large-zh-v1.5'
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.embedding_model = "transformers"
                
                # 测试获取向量维度
                inputs = self.tokenizer("测试文本", return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                    self.vector_dimension = len(embedding)
                
                print(f"✅ 使用transformers模型: {model_name}，向量维度: {self.vector_dimension}")
                return
            except ImportError:
                logger.warning("transformers未安装，建议安装: pip install transformers torch")
            except Exception as e:
                logger.warning(f"无法加载transformers模型: {e}")
    
            # 如果所有方案都失败，使用模拟向量
            print("⚠️ 无法加载任何embedding模型，将使用模拟向量")
            print("⚠️ 建议安装: pip install sentence-transformers")
            print("⚠️ 或配置RAGFlow的embedding模型")
            self.embedding_model = None
            self.vector_dimension = 1024  # jina-embeddings-v3的向量维度
            
        except Exception as e:
            logger.error(f"❌ embedding模型初始化失败: {e}")
            self.embedding_model = None
            self.vector_dimension = 1024  # jina-embeddings-v3的向量维度
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成文本的embedding向量（与simple_kb_setup.py保持一致）"""
        try:
            if self.embedding_model is None:
                # 使用模拟向量作为备选
                hash_value = hash(text) % (2**32)
                np.random.seed(hash_value)
                vector = np.random.normal(0, 1, self.vector_dimension)
                vector = vector / np.linalg.norm(vector)
                return vector.tolist()
            
            # OpenAI客户端（Xinference）
            if hasattr(self.embedding_model, 'embeddings'):
                response = self.embedding_model.embeddings.create(
                    input=[text],
                    model="jina-embeddings-v3"
                )
                return response.data[0].embedding
            
            # RAGFlow LLMBundle
            elif hasattr(self.embedding_model, 'encode') and hasattr(self.embedding_model, 'mdl'):
                embeddings, _ = self.embedding_model.encode([text])
                return embeddings[0]
            
            # RAGFlow embedding模型类
            elif hasattr(self.embedding_model, 'encode') and not hasattr(self.embedding_model, 'mdl'):
                embeddings, _ = self.embedding_model.encode([text])
                return embeddings[0]
            
            # sentence-transformers
            elif hasattr(self.embedding_model, 'encode'):
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
            vector = np.random.normal(0, 1, self.vector_dimension)
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()
    
    def create_simple_search(self, query_text):
        """创建简单的文本搜索表达式"""
        return [MatchTextExpr(
            fields=["content_ltks^10", "title_tks^8", "content_with_weight^2"],
            matching_text=query_text,
            topn=10,
            extra_options={"minimum_should_match": 0.3}
        )]
    
    def create_vector_search(self, query_text):
        """创建向量搜索表达式（使用真实embedding）"""
        # 生成查询向量
        query_vector = self.generate_embedding(query_text)
        print(f"  查询向量维度: {len(query_vector)}")
        
        # 动态生成向量字段名
        vector_field_name = f"q_{self.vector_dimension}_vec"
        
        return [MatchDenseExpr(
            vector_column_name=vector_field_name,
            embedding_data=query_vector,
            topn=10,
            extra_options={"similarity": 0.1}
        )]
    
    def create_hybrid_search(self, query_text, vector_weight=0.5):
        """创建混合搜索表达式（文本+真实向量）"""
        expressions = []
        
        # 文本搜索
        text_expr = MatchTextExpr(
            fields=["content_ltks^10", "title_tks^8", "content_with_weight^2"],
            matching_text=query_text,
            topn=10,
            extra_options={"minimum_should_match": 0.3}
        )
        expressions.append(text_expr)
        
        # 向量搜索（使用真实embedding）
        query_vector = self.generate_embedding(query_text)
        vector_field_name = f"q_{self.vector_dimension}_vec"
        vector_expr = MatchDenseExpr(
            vector_column_name=vector_field_name,
            embedding_data=query_vector,
            topn=10,
            extra_options={"similarity": 0.1}
        )
        expressions.append(vector_expr)
        
        # 融合表达式
        text_weight = 1.0 - vector_weight
        fusion_expr = FusionExpr(
            method="weighted_sum",
            topn=10,
            fusion_params={"weights": f"{text_weight}, {vector_weight}"}
        )
        expressions.append(fusion_expr)
        
        return expressions

def test_basic_search():
    """测试基本搜索功能"""
    print("🔍 快速搜索引擎对比测试（使用真实embedding向量）")
    print("=" * 60)
    
    # 尝试加载自动生成的配置
    try:
        from test_kb_config import TEST_CONFIG
        INDEX_NAME = TEST_CONFIG["index_name"]
        KB_IDS = TEST_CONFIG["kb_ids"]
        TEST_QUERIES = TEST_CONFIG["test_queries"][:3]  # 使用前3个测试查询
        print("✅ 使用自动生成的配置")
    except ImportError:
        # 如果没有配置文件，使用默认值
        print("⚠️ 未找到test_kb_config.py，使用默认配置")
        print("💡 建议先运行 python simple_kb_setup.py 创建测试数据")
        INDEX_NAME = "ragflow_6a2a5a8c00a611f0883a0242ac140006"
        KB_IDS = ["6a2a5a8c-00a6-11f0-883a-0242ac140006"]
        TEST_QUERIES = ["人工智能", "机器学习", "自然语言处理"]
    
    print(f"索引名称: {INDEX_NAME}")
    print(f"知识库ID: {KB_IDS}")
    print(f"测试查询: {TEST_QUERIES}")
    print()
    
    # 初始化测试器
    tester = QuickSearchTester()
    
    if not tester.es_conn and not tester.pd_conn:
        print("❌ 没有可用的搜索引擎连接")
        return
    
    print(f"Embedding模型: {type(tester.embedding_model).__name__ if tester.embedding_model else 'simulated'}")
    print()
    
    # 测试每个查询
    for query_idx, test_query in enumerate(TEST_QUERIES):
        print(f"🧪 测试查询 {query_idx + 1}/{len(TEST_QUERIES)}: '{test_query}'")
        print("=" * 50)
    
    # 测试不同的搜索方式
    test_cases = [
            ("纯文本搜索", tester.create_simple_search(test_query)),
            ("纯向量搜索", tester.create_vector_search(test_query)),
            ("混合搜索(0.5)", tester.create_hybrid_search(test_query, 0.5))
    ]
    
    for test_name, match_exprs in test_cases:
            print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        # ES搜索
            if tester.es_conn:
            try:
                start_time = time.time()
                    es_res = tester.es_conn.search(
                        selectFields=["id", "title_tks", "content_with_weight"],
                    highlightFields=[],
                    condition={"available_int": 1},
                    matchExprs=match_exprs,
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=5,
                    indexNames=INDEX_NAME,
                    knowledgebaseIds=KB_IDS
                )
                es_time = time.time() - start_time
                
                es_hits = es_res.get("hits", {}).get("hits", [])
                print(f"ES结果: {len(es_hits)}条, 耗时{es_time:.3f}秒")
                
                for i, hit in enumerate(es_hits[:3]):
                        source = hit.get("_source", {})
                        title = source.get("title_tks", "")
                    score = hit.get("_score", 0)
                        print(f"  {i+1}. {title} (分数: {score:.3f})")
                    
            except Exception as e:
                print(f"ES搜索失败: {e}")
        
        # ParadeDB搜索
            if tester.pd_conn:
            try:
                start_time = time.time()
                    pd_res = tester.pd_conn.search(
                        selectFields=["id", "title_tks", "content_with_weight"],
                    highlightFields=[],
                    condition={"available_int": 1},
                    matchExprs=match_exprs,
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=5,
                    indexNames=INDEX_NAME,
                    knowledgebaseIds=KB_IDS
                )
                pd_time = time.time() - start_time
                
                pd_hits = pd_res.get("hits", {}).get("hits", [])
                print(f"ParadeDB结果: {len(pd_hits)}条, 耗时{pd_time:.3f}秒")
                
                for i, hit in enumerate(pd_hits[:3]):
                        source = hit.get("_source", {})
                        title = source.get("title_tks", "")
                    score = hit.get("_score", 0)
                        print(f"  {i+1}. {title} (分数: {score:.3f})")
                    
            except Exception as e:
                print(f"ParadeDB搜索失败: {e}")
        
        # 简单对比
            if tester.es_conn and tester.pd_conn:
            try:
                es_hits = es_res.get("hits", {}).get("hits", [])
                pd_hits = pd_res.get("hits", {}).get("hits", [])
                
                    if es_hits and pd_hits:
                        # 提取文档ID进行对比
                        es_ids = [hit.get("_id") for hit in es_hits]
                        pd_ids = [hit.get("_id") for hit in pd_hits]
                        
                        # 计算重叠
                        common_ids = set(es_ids) & set(pd_ids)
                        overlap_ratio = len(common_ids) / max(len(es_ids), len(pd_ids)) if max(len(es_ids), len(pd_ids)) > 0 else 0
                
                        print(f"📊 结果对比: 重叠率 {overlap_ratio:.2%} ({len(common_ids)}/{max(len(es_ids), len(pd_ids))})")
                
            except Exception as e:
                    print(f"结果对比失败: {e}")
        
        print()  # 查询间的分隔
    
    print("🎉 快速测试完成!")
    print("💡 如需详细分析，请运行: python compare_search_engines.py")

def main():
    """主函数"""
    try:
        test_basic_search()
        print("✅ 快速测试完成")
        print("\n💡 提示: 如需详细分析，请使用 compare_search_engines.py")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 