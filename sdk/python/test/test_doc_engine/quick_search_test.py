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
        """初始化embedding模型"""
        try:
            # 方法1: 直接使用OpenAI客户端调用Xinference
            try:
                from openai import OpenAI
                
                xinference_client = OpenAI(
                    api_key="empty", 
                    base_url="http://120.77.38.66:8008/v1"
                )
                
                test_response = xinference_client.embeddings.create(
                    input=["测试文本"],
                    model="jina-embeddings-v3"
                )
                
                if test_response.data and len(test_response.data[0].embedding) > 0:
                    self.embedding_model = xinference_client
                    self.vector_dimension = len(test_response.data[0].embedding)
                    print(f"✅ 使用OpenAI客户端调用Xinference jina-embeddings-v3模型，向量维度: {self.vector_dimension}")
                    return
                    
            except Exception as e:
                logger.warning(f"无法使用OpenAI客户端调用Xinference: {e}")
            
            # 如果所有方案都失败，使用模拟向量
            logger.info("⚠️ 无法加载任何真实embedding模型，使用模拟向量进行测试")
            self.embedding_model = None
            self.vector_dimension = 1024
            
        except Exception as e:
            logger.error(f"❌ embedding模型初始化失败: {e}")
            self.embedding_model = None
            self.vector_dimension = 1024
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成文本的embedding向量"""
        try:
            if self.embedding_model is None:
                # 使用模拟向量
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
            
            # 其他类型的embedding模型
            elif hasattr(self.embedding_model, 'encode'):
                if hasattr(self.embedding_model, 'mdl'):
                    # RAGFlow LLMBundle
                    embeddings, _ = self.embedding_model.encode([text])
                    return embeddings[0]
                else:
                    # sentence-transformers
                    embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                    return embedding.tolist()
            
            else:
                raise Exception("未知的embedding模型类型")
                
        except Exception as e:
            logger.warning(f"生成embedding失败: {e}，使用模拟向量")
            hash_value = hash(text) % (2**32)
            np.random.seed(hash_value)
            vector = np.random.normal(0, 1, self.vector_dimension)
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()
    
    def create_simple_search(self, query_text):
        """创建简单文本搜索"""
        return [MatchTextExpr(
            fields=["title_tks^10", "title_sm_tks^5", "important_kwd^30", "important_tks^20", "question_tks^20", "content_ltks^2", "content_sm_ltks"],
            matching_text=query_text,
            topn=10
        )]
    
    def create_vector_search(self, query_text):
        """创建向量搜索"""
        vector = self.generate_embedding(query_text)
        vector_field = f"q_{self.vector_dimension}_vec"
        return [MatchDenseExpr(
            vector_column_name=vector_field,
            embedding_data=vector,
            embedding_data_type="float",
            distance_type="cosine",
            topn=10
        )]
    
    def create_hybrid_search(self, query_text, vector_weight=0.5):
        """创建混合搜索（文本+向量）"""
        vector = self.generate_embedding(query_text)
        vector_field = f"q_{self.vector_dimension}_vec"
        
        text_weight = 1.0 - vector_weight
        
        return [
            MatchTextExpr(
                fields=["title_tks^10", "title_sm_tks^5", "important_kwd^30", "important_tks^20", "question_tks^20", "content_ltks^2", "content_sm_ltks"],
                matching_text=query_text,
                topn=10
            ),
            MatchDenseExpr(
                vector_column_name=vector_field,
                embedding_data=vector,
                embedding_data_type="float",
                distance_type="cosine",
                topn=10
            ),
            FusionExpr(
                method="weighted_sum",
                topn=10,
                fusion_params={"weights": f"{text_weight:.3f}, {vector_weight:.3f}"}
            )
        ]
    
    def run_search(self, conn, index_name, kb_ids, match_exprs, search_type=""):
        """执行搜索并返回结果"""
        try:
            start_time = time.time()
            
            results = conn.search(
                selectFields=["id", "title_tks", "content_with_weight"],
                highlightFields=[],
                condition={"available_int": 1},
                matchExprs=match_exprs,
                orderBy=OrderByExpr(),
                offset=0,
                limit=5,
                indexNames=index_name,
                knowledgebaseIds=kb_ids
            )
            
            end_time = time.time()
            search_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            total_hits = conn.getTotal(results)
            hits = results.get("hits", {}).get("hits", [])
            
            return {
                "total": total_hits,
                "hits": hits,
                "time_ms": search_time,
                "search_type": search_type
            }
            
        except Exception as e:
            logger.error(f"搜索失败 ({search_type}): {e}")
            return {
                "total": 0,
                "hits": [],
                "time_ms": 0,
                "search_type": search_type,
                "error": str(e)
            }
    
    def format_results(self, results, engine_name):
        """格式化搜索结果"""
        if "error" in results:
            return f"❌ {engine_name} 搜索失败: {results['error']}"
        
        output = []
        output.append(f"🔍 {engine_name} ({results['search_type']}):")
        output.append(f"   总数: {results['total']}, 耗时: {results['time_ms']:.1f}ms")
        
        for i, hit in enumerate(results['hits'][:3], 1):
            source = hit.get('_source', {})
            title = source.get('title_tks', '无标题')
            content = source.get('content_with_weight', '')[:100]
            output.append(f"   {i}. {title}")
            output.append(f"      {content}...")
        
        return "\n".join(output)
    
    def test_search_query(self, query_text, index_name, kb_ids):
        """测试单个查询"""
        print(f"\n🔍 测试查询: '{query_text}'")
        print("=" * 60)
        
        # 1. 文本搜索
        print("\n📝 文本搜索:")
        text_match_exprs = self.create_simple_search(query_text)
        
        if self.es_conn:
            es_text_results = self.run_search(self.es_conn, index_name, kb_ids, text_match_exprs, "文本搜索")
            print(self.format_results(es_text_results, "Elasticsearch"))
        
        if self.pd_conn:
            pd_text_results = self.run_search(self.pd_conn, index_name, kb_ids, text_match_exprs, "文本搜索")
            print(self.format_results(pd_text_results, "ParadeDB"))
        
        # 2. 向量搜索
        print(f"\n🎯 向量搜索:")
        vector_match_exprs = self.create_vector_search(query_text)
        
        if self.es_conn:
            es_vector_results = self.run_search(self.es_conn, index_name, kb_ids, vector_match_exprs, "向量搜索")
            print(self.format_results(es_vector_results, "Elasticsearch"))
        
        if self.pd_conn:
            pd_vector_results = self.run_search(self.pd_conn, index_name, kb_ids, vector_match_exprs, "向量搜索")
            print(self.format_results(pd_vector_results, "ParadeDB"))
        
        # 3. 混合搜索
        print(f"\n🔀 混合搜索 (向量权重=0.7):")
        hybrid_match_exprs = self.create_hybrid_search(query_text, vector_weight=0.7)
        
        if self.es_conn:
            es_hybrid_results = self.run_search(self.es_conn, index_name, kb_ids, hybrid_match_exprs, "混合搜索")
            print(self.format_results(es_hybrid_results, "Elasticsearch"))
        
        if self.pd_conn:
            pd_hybrid_results = self.run_search(self.pd_conn, index_name, kb_ids, hybrid_match_exprs, "混合搜索")
            print(self.format_results(pd_hybrid_results, "ParadeDB"))

def main():
    """主函数"""
    print("🚀 快速搜索引擎对比测试")
    print("=" * 50)
    
    # 初始化测试器
    tester = QuickSearchTester()
    
    # 检查连接状态
    if not tester.es_conn and not tester.pd_conn:
        print("❌ 没有可用的搜索引擎连接")
        return
    
    # 尝试加载测试配置
    try:
        from test_kb_config import TEST_CONFIG
        index_name = TEST_CONFIG["index_name"]
        kb_ids = TEST_CONFIG["kb_ids"]
        test_queries = TEST_CONFIG["test_queries"][:5]  # 只测试前5个查询
        print(f"✅ 加载测试配置成功")
        print(f"   索引: {index_name}")
        print(f"   知识库ID: {kb_ids}")
    except ImportError:
        print("⚠️ 未找到test_kb_config.py，请先运行simple_kb_setup.py")
        return
    
    # 执行测试查询
    for query in test_queries:
        tester.test_search_query(query, index_name, kb_ids)
        time.sleep(1)  # 避免请求过于频繁
    
    print(f"\n🎉 测试完成!")
    print(f"💡 如需更详细的对比分析，请运行: python compare_search_engines.py")

if __name__ == "__main__":
    main() 