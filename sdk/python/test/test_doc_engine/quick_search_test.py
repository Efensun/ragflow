#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速搜索引擎对比测试脚本

这是一个简化版本，用于快速验证ES和ParadeDB的基本搜索功能
"""

import os
import sys
import time
import logging

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

def test_basic_search():
    """测试基本搜索功能"""
    print("🔍 快速搜索引擎对比测试")
    print("=" * 40)
    
    # 尝试加载自动生成的配置
    try:
        from test_kb_config import TEST_CONFIG
        INDEX_NAME = TEST_CONFIG["index_name"]
        KB_IDS = TEST_CONFIG["kb_ids"]
        TEST_QUERY = TEST_CONFIG["test_queries"][0]  # 使用第一个测试查询
        print("✅ 使用自动生成的配置")
    except ImportError:
        # 如果没有配置文件，使用默认值
        print("⚠️ 未找到test_kb_config.py，使用默认配置")
        print("💡 建议先运行 python simple_kb_setup.py 创建测试数据")
        INDEX_NAME = "ragflow_6a2a5a8c00a611f0883a0242ac140006"
        KB_IDS = ["6a2a5a8c-00a6-11f0-883a-0242ac140006"]
        TEST_QUERY = "机器人"
    
    print(f"索引名称: {INDEX_NAME}")
    print(f"知识库ID: {KB_IDS}")
    print(f"测试查询: '{TEST_QUERY}'")
    print()
    
    # 初始化连接
    es_conn = None
    pd_conn = None
    
    try:
        # 尝试连接ES
        if hasattr(settings, 'ES') and settings.ES.get('hosts'):
            print("🔗 连接Elasticsearch...")
            es_conn = ESConnection()
            print("✅ ES连接成功")
        else:
            print("⚠️ 未配置ES，跳过ES测试")
    except Exception as e:
        print(f"❌ ES连接失败: {e}")
    
    try:
        # 尝试连接ParadeDB
        if hasattr(settings, 'PARADEDB') and settings.PARADEDB.get('host'):
            print("🔗 连接ParadeDB...")
            pd_conn = PDConnection()
            print("✅ ParadeDB连接成功")
        else:
            print("⚠️ 未配置ParadeDB，跳过ParadeDB测试")
    except Exception as e:
        print(f"❌ ParadeDB连接失败: {e}")
    
    if not es_conn and not pd_conn:
        print("❌ 没有可用的搜索引擎连接")
        return
    
    print()
    
    # 创建搜索表达式
    def create_simple_search(query_text):
        """创建简单的文本搜索表达式"""
        return [MatchTextExpr(
            fields=["content_ltks^10", "title_tks^8", "content_with_weight^2"],
            matching_text=query_text,
            topn=10,
            extra_options={"minimum_should_match": 0.3}
        )]
    
    def create_hybrid_search(query_text):
        """创建混合搜索表达式"""
        expressions = []
        
        # 文本搜索
        text_expr = MatchTextExpr(
            fields=["content_ltks^10", "title_tks^8", "content_with_weight^2"],
            matching_text=query_text,
            topn=10,
            extra_options={"minimum_should_match": 0.3}
        )
        expressions.append(text_expr)
        
        # 向量搜索（使用随机向量进行测试）
        np.random.seed(42)
        test_vector = np.random.normal(0, 1, 1024)
        test_vector = test_vector / np.linalg.norm(test_vector)
        
        vector_expr = MatchDenseExpr(
            vector_column_name="q_1024_vec",
            embedding_data=test_vector.tolist(),
            topn=10,
            extra_options={"similarity": 0.1}
        )
        expressions.append(vector_expr)
        
        # 融合表达式
        fusion_expr = FusionExpr(
            method="weighted_sum",
            topn=10,
            fusion_params={"weights": "0.5, 0.5"}
        )
        expressions.append(fusion_expr)
        
        return expressions
    
    # 测试不同的搜索方式
    test_cases = [
        ("纯文本搜索", create_simple_search(TEST_QUERY)),
        ("混合搜索", create_hybrid_search(TEST_QUERY))
    ]
    
    for test_name, match_exprs in test_cases:
        print(f"🧪 测试: {test_name}")
        print("-" * 30)
        
        # ES搜索
        if es_conn:
            try:
                start_time = time.time()
                es_res = es_conn.search(
                    selectFields=["id", "content_with_weight"],
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
                    doc_id = hit.get("_id", "")[:20] + "..."
                    score = hit.get("_score", 0)
                    print(f"  {i+1}. {doc_id} (分数: {score:.3f})")
                    
            except Exception as e:
                print(f"ES搜索失败: {e}")
        
        # ParadeDB搜索
        if pd_conn:
            try:
                start_time = time.time()
                pd_res = pd_conn.search(
                    selectFields=["id", "content_with_weight"],
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
                    doc_id = hit.get("_id", "")[:20] + "..."
                    score = hit.get("_score", 0)
                    print(f"  {i+1}. {doc_id} (分数: {score:.3f})")
                    
            except Exception as e:
                print(f"ParadeDB搜索失败: {e}")
        
        # 简单对比
        if es_conn and pd_conn:
            try:
                es_hits = es_res.get("hits", {}).get("hits", [])
                pd_hits = pd_res.get("hits", {}).get("hits", [])
                
                es_ids = set(hit.get("_id", "") for hit in es_hits)
                pd_ids = set(hit.get("_id", "") for hit in pd_hits)
                
                common_docs = len(es_ids & pd_ids)
                total_docs = max(len(es_ids), len(pd_ids))
                overlap_rate = common_docs / total_docs if total_docs > 0 else 0
                
                print(f"📊 快速对比:")
                print(f"  共同文档: {common_docs}/{total_docs}")
                print(f"  重叠率: {overlap_rate*100:.1f}%")
                
                # 检查Top-1是否相同
                if es_hits and pd_hits:
                    es_top1 = es_hits[0].get("_id", "")
                    pd_top1 = pd_hits[0].get("_id", "")
                    top1_match = "✅" if es_top1 == pd_top1 else "❌"
                    print(f"  Top-1匹配: {top1_match}")
                
            except Exception as e:
                print(f"对比分析失败: {e}")
        
        print()

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