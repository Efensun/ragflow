#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES和ParadeDB搜索结果对比验证脚本

该脚本用于验证在相同配置下，ES和ParadeDB是否能召回大致相同顺序的文档块，
以及相似度的一致性。

主要功能：
1. 使用相同的查询参数分别调用ES和ParadeDB
2. 比较返回结果的文档ID顺序
3. 分析相似度分数的差异
4. 生成详细的对比报告
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import settings
from rag.utils.es_conn import ESConnection
from rag.utils.pd_conn import PDConnection
from rag.utils.doc_store_conn import MatchTextExpr, MatchDenseExpr, FusionExpr, OrderByExpr
from rag.nlp import rag_tokenizer
import hashlib

# 导入配置
try:
    from search_comparison_config import get_config, validate_config
    CONFIG = get_config()
except ImportError:
    print("⚠️ 未找到配置文件 search_comparison_config.py，使用默认配置")
    CONFIG = None

# 配置日志
log_config = CONFIG['logging'] if CONFIG else {'level': 'INFO', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
logging.basicConfig(
    level=getattr(logging, log_config['level']),
    format=log_config['format']
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据结构"""
    doc_id: str
    score: float
    content_preview: str
    source: str  # 'es' or 'paradedb'
    rank: int

@dataclass
class ComparisonMetrics:
    """比较指标"""
    total_es_results: int
    total_pd_results: int
    common_docs: int
    rank_correlation: float
    score_correlation: float
    avg_score_diff: float
    top_k_overlap: Dict[int, float]  # k -> overlap_ratio

class SearchEngineComparator:
    """搜索引擎比较器"""
    
    def __init__(self):
        self.es_conn = None
        self.pd_conn = None
        self.vector_dimension = 1024  # 默认向量维度
        self.init_connections()
        self.detect_vector_dimension()
    
    def init_connections(self):
        """初始化数据库连接"""
        try:
            # 初始化ES连接
            if hasattr(settings, 'ES') and settings.ES.get('hosts'):
                logger.info("初始化Elasticsearch连接...")
                self.es_conn = ESConnection()
                logger.info("✅ Elasticsearch连接成功")
            else:
                logger.warning("⚠️ 未配置Elasticsearch，跳过ES测试")
            
            # 初始化ParadeDB连接
            if hasattr(settings, 'PARADEDB') and settings.PARADEDB.get('host'):
                logger.info("初始化ParadeDB连接...")
                self.pd_conn = PDConnection()
                logger.info("✅ ParadeDB连接成功")
            else:
                logger.warning("⚠️ 未配置ParadeDB，跳过ParadeDB测试")
                
        except Exception as e:
            logger.error(f"❌ 初始化数据库连接失败: {e}")
            raise
    
    def detect_vector_dimension(self):
        """自动检测向量维度"""
        try:
            # 尝试从配置文件中获取向量维度
            try:
                from test_kb_config import TEST_CONFIG
                # 从测试配置中获取一个样本文档的向量维度
                index_name = TEST_CONFIG["index_name"]
                kb_ids = TEST_CONFIG["kb_ids"]
                
                # 尝试从ES获取样本文档
                if self.es_conn:
                    from rag.utils.doc_store_conn import OrderByExpr
                    res = self.es_conn.search(
                        selectFields=["metadata"],
                        highlightFields=[],
                        condition={"available_int": 1},
                        matchExprs=[],
                        orderBy=OrderByExpr(),
                        offset=0,
                        limit=1,
                        indexNames=index_name,
                        knowledgebaseIds=kb_ids
                    )
                    hits = res.get("hits", {}).get("hits", [])
                    if hits:
                        metadata = hits[0].get("_source", {}).get("metadata", {})
                        if "vector_dimension" in metadata:
                            self.vector_dimension = metadata["vector_dimension"]
                            logger.info(f"✅ 从ES样本文档检测到向量维度: {self.vector_dimension}")
                            return
                
                # 如果ES没有找到，尝试从ParadeDB获取
                if self.pd_conn:
                    from rag.utils.doc_store_conn import OrderByExpr
                    res = self.pd_conn.search(
                        selectFields=["metadata"],
                        highlightFields=[],
                        condition={"available_int": 1},
                        matchExprs=[],
                        orderBy=OrderByExpr(),
                        offset=0,
                        limit=1,
                        indexNames=index_name,
                        knowledgebaseIds=kb_ids
                    )
                    hits = res.get("hits", {}).get("hits", [])
                    if hits:
                        metadata = hits[0].get("_source", {}).get("metadata", {})
                        if "vector_dimension" in metadata:
                            self.vector_dimension = metadata["vector_dimension"]
                            logger.info(f"✅ 从ParadeDB样本文档检测到向量维度: {self.vector_dimension}")
                            return
                            
            except ImportError:
                logger.info("未找到test_kb_config.py，使用默认向量维度")
            except Exception as e:
                logger.warning(f"从配置文件检测向量维度失败: {e}")
            
            logger.info(f"使用默认向量维度: {self.vector_dimension}")
            
        except Exception as e:
            logger.warning(f"向量维度检测失败: {e}，使用默认值: {self.vector_dimension}")
    
    def generate_test_vector(self, dimension: int = 1024) -> List[float]:
        """生成测试向量"""
        # 使用固定种子确保可重复性
        np.random.seed(42)
        vector = np.random.normal(0, 1, dimension)
        # 归一化向量
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def create_search_expressions(self, query_text: str, vector_dim: int = 1024, 
                                vector_weight: float = 0.5) -> List:
        """创建搜索表达式"""
        expressions = []
        
        # 文本搜索表达式 - 使用配置文件中的字段设置
        if CONFIG and 'test' in CONFIG and 'search_fields' in CONFIG['test']:
            text_fields = CONFIG['test']['search_fields']
        else:
            text_fields = [
                "content_ltks^10",
                "content_sm_ltks^5", 
                "title_tks^8",
                "title_sm_tks^4",
                "important_tks^6",
                "question_tks^3",
                "content_with_weight^2"
            ]
        
        match_text = MatchTextExpr(
            fields=text_fields,
            matching_text=query_text,
            topn=50,
            extra_options={"minimum_should_match": 0.3}
        )
        expressions.append(match_text)
        
        # 向量搜索表达式
        test_vector = self.generate_test_vector(vector_dim)
        match_dense = MatchDenseExpr(
            vector_column_name=f"q_{vector_dim}_vec",
            embedding_data=test_vector,
            topn=50,
            extra_options={"similarity": 0.1}
        )
        expressions.append(match_dense)
        
        # 融合表达式
        fusion_expr = FusionExpr(
            method="weighted_sum",
            topn=50,
            fusion_params={
                "weights": f"{1.0 - vector_weight}, {vector_weight}"
            }
        )
        expressions.append(fusion_expr)
        
        return expressions
    
    def search_with_engine(self, engine_name: str, conn, index_name: str, 
                          kb_ids: List[str], query_text: str, 
                          vector_weight: float = 0.5, limit: int = 20) -> List[SearchResult]:
        """使用指定引擎进行搜索"""
        try:
            # 创建搜索表达式，使用检测到的向量维度
            match_exprs = self.create_search_expressions(query_text, self.vector_dimension, vector_weight)
            
            # 搜索参数 - 使用配置文件中的设置
            if CONFIG and 'test' in CONFIG:
                select_fields = CONFIG['test'].get('select_fields', ["id", "content_with_weight", "title_tks", "docnm_kwd"])
                highlight_fields = CONFIG['test'].get('highlight_fields', ["content_ltks", "title_tks"])
            else:
                select_fields = ["id", "content_with_weight", "title_tks", "docnm_kwd"]
                highlight_fields = ["content_ltks", "title_tks"]
            condition = {"available_int": 1}
            order_by = OrderByExpr()
            
            # 执行搜索
            start_time = time.time()
            res = conn.search(
                selectFields=select_fields,
                highlightFields=highlight_fields,
                condition=condition,
                matchExprs=match_exprs,
                orderBy=order_by,
                offset=0,
                limit=limit,
                indexNames=index_name,
                knowledgebaseIds=kb_ids
            )
            search_time = time.time() - start_time
            
            # 解析结果
            results = []
            hits = res.get("hits", {}).get("hits", [])
            
            for rank, hit in enumerate(hits):
                doc_id = hit.get("_id", "")
                score = hit.get("_score", 0.0)
                source_data = hit.get("_source", {})
                
                # 获取内容预览
                content = source_data.get("content_with_weight", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                
                result = SearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    content_preview=content,
                    source=engine_name,
                    rank=rank + 1
                )
                results.append(result)
            
            logger.info(f"{engine_name}搜索完成: {len(results)}条结果, 耗时{search_time:.3f}秒")
            return results
            
        except Exception as e:
            logger.error(f"❌ {engine_name}搜索失败: {e}")
            return []
    
    def calculate_rank_correlation(self, es_results: List[SearchResult], 
                                 pd_results: List[SearchResult]) -> float:
        """计算排名相关性（Spearman相关系数）"""
        # 获取共同文档
        es_docs = {r.doc_id: r.rank for r in es_results}
        pd_docs = {r.doc_id: r.rank for r in pd_results}
        
        common_docs = set(es_docs.keys()) & set(pd_docs.keys())
        
        if len(common_docs) < 2:
            return 0.0
        
        # 提取排名
        es_ranks = [es_docs[doc_id] for doc_id in common_docs]
        pd_ranks = [pd_docs[doc_id] for doc_id in common_docs]
        
        # 计算Spearman相关系数
        try:
            correlation = np.corrcoef(es_ranks, pd_ranks)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def calculate_score_correlation(self, es_results: List[SearchResult], 
                                  pd_results: List[SearchResult]) -> Tuple[float, float]:
        """计算分数相关性和平均差异"""
        # 获取共同文档的分数
        es_scores = {r.doc_id: r.score for r in es_results}
        pd_scores = {r.doc_id: r.score for r in pd_results}
        
        common_docs = set(es_scores.keys()) & set(pd_scores.keys())
        
        if len(common_docs) < 2:
            return 0.0, 0.0
        
        es_score_list = [es_scores[doc_id] for doc_id in common_docs]
        pd_score_list = [pd_scores[doc_id] for doc_id in common_docs]
        
        # 计算相关性
        try:
            correlation = np.corrcoef(es_score_list, pd_score_list)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0.0
        except:
            correlation = 0.0
        
        # 计算平均差异
        score_diffs = [abs(es_scores[doc_id] - pd_scores[doc_id]) for doc_id in common_docs]
        avg_diff = np.mean(score_diffs) if score_diffs else 0.0
        
        return correlation, avg_diff
    
    def calculate_top_k_overlap(self, es_results: List[SearchResult], 
                               pd_results: List[SearchResult]) -> Dict[int, float]:
        """计算Top-K重叠率"""
        overlaps = {}
        
        for k in [1, 3, 5, 10, 20]:
            if k > min(len(es_results), len(pd_results)):
                continue
                
            es_top_k = set(r.doc_id for r in es_results[:k])
            pd_top_k = set(r.doc_id for r in pd_results[:k])
            
            overlap = len(es_top_k & pd_top_k)
            overlap_ratio = overlap / k
            overlaps[k] = overlap_ratio
        
        return overlaps
    
    def compare_search_results(self, es_results: List[SearchResult], 
                             pd_results: List[SearchResult]) -> ComparisonMetrics:
        """比较搜索结果"""
        # 基本统计
        total_es = len(es_results)
        total_pd = len(pd_results)
        
        es_doc_ids = set(r.doc_id for r in es_results)
        pd_doc_ids = set(r.doc_id for r in pd_results)
        common_docs = len(es_doc_ids & pd_doc_ids)
        
        # 计算各种指标
        rank_corr = self.calculate_rank_correlation(es_results, pd_results)
        score_corr, avg_score_diff = self.calculate_score_correlation(es_results, pd_results)
        top_k_overlap = self.calculate_top_k_overlap(es_results, pd_results)
        
        return ComparisonMetrics(
            total_es_results=total_es,
            total_pd_results=total_pd,
            common_docs=common_docs,
            rank_correlation=rank_corr,
            score_correlation=score_corr,
            avg_score_diff=avg_score_diff,
            top_k_overlap=top_k_overlap
        )
    
    def print_detailed_comparison(self, query_text: str, es_results: List[SearchResult], 
                                pd_results: List[SearchResult], metrics: ComparisonMetrics):
        """打印详细的比较结果"""
        print(f"\n{'='*80}")
        print(f"查询: '{query_text}'")
        print(f"{'='*80}")
        
        # 基本统计
        print(f"\n📊 基本统计:")
        print(f"  ES结果数量: {metrics.total_es_results}")
        print(f"  ParadeDB结果数量: {metrics.total_pd_results}")
        print(f"  共同文档数量: {metrics.common_docs}")
        print(f"  文档重叠率: {metrics.common_docs/max(metrics.total_es_results, metrics.total_pd_results, 1)*100:.1f}%")
        
        # 相关性指标
        print(f"\n🔗 相关性指标:")
        print(f"  排名相关性 (Spearman): {metrics.rank_correlation:.3f}")
        print(f"  分数相关性 (Pearson): {metrics.score_correlation:.3f}")
        print(f"  平均分数差异: {metrics.avg_score_diff:.3f}")
        
        # Top-K重叠率
        print(f"\n🎯 Top-K重叠率:")
        for k, overlap in metrics.top_k_overlap.items():
            print(f"  Top-{k}: {overlap*100:.1f}%")
        
        # 详细结果对比（前10条）
        print(f"\n📋 详细结果对比 (前10条):")
        print(f"{'排名':<4} {'ES文档ID':<20} {'ES分数':<8} {'PD文档ID':<20} {'PD分数':<8} {'匹配':<4}")
        print("-" * 80)
        
        max_results = min(10, max(len(es_results), len(pd_results)))
        for i in range(max_results):
            es_doc = es_results[i] if i < len(es_results) else None
            pd_doc = pd_results[i] if i < len(pd_results) else None
            
            es_id = es_doc.doc_id[:18] + ".." if es_doc and len(es_doc.doc_id) > 20 else (es_doc.doc_id if es_doc else "-")
            es_score = f"{es_doc.score:.3f}" if es_doc else "-"
            
            pd_id = pd_doc.doc_id[:18] + ".." if pd_doc and len(pd_doc.doc_id) > 20 else (pd_doc.doc_id if pd_doc else "-")
            pd_score = f"{pd_doc.score:.3f}" if pd_doc else "-"
            
            match = "✓" if es_doc and pd_doc and es_doc.doc_id == pd_doc.doc_id else "✗"
            
            print(f"{i+1:<4} {es_id:<20} {es_score:<8} {pd_id:<20} {pd_score:<8} {match:<4}")
    
    def run_comparison_test(self, index_name: str, kb_ids: List[str], 
                           test_queries: List[str], vector_weights: List[float] = [0.3, 0.5, 0.7]):
        """运行完整的比较测试"""
        print(f"\n🚀 开始搜索引擎比较测试")
        print(f"索引名称: {index_name}")
        print(f"知识库ID: {kb_ids}")
        print(f"测试查询数量: {len(test_queries)}")
        print(f"向量权重配置: {vector_weights}")
        
        if not self.es_conn and not self.pd_conn:
            logger.error("❌ 没有可用的搜索引擎连接")
            return
        
        all_metrics = []
        
        for query_idx, query_text in enumerate(test_queries):
            print(f"\n🔍 测试查询 {query_idx + 1}/{len(test_queries)}: '{query_text}'")
            
            for weight_idx, vector_weight in enumerate(vector_weights):
                print(f"\n  📊 向量权重: {vector_weight}")
                
                es_results = []
                pd_results = []
                
                # ES搜索
                if self.es_conn:
                    try:
                        es_results = self.search_with_engine(
                            "ES", self.es_conn, index_name, kb_ids, 
                            query_text, vector_weight, limit=20
                        )
                    except Exception as e:
                        logger.error(f"ES搜索失败: {e}")
                
                # ParadeDB搜索
                if self.pd_conn:
                    try:
                        pd_results = self.search_with_engine(
                            "ParadeDB", self.pd_conn, index_name, kb_ids, 
                            query_text, vector_weight, limit=20
                        )
                    except Exception as e:
                        logger.error(f"ParadeDB搜索失败: {e}")
                
                # 比较结果
                if es_results and pd_results:
                    metrics = self.compare_search_results(es_results, pd_results)
                    all_metrics.append({
                        'query': query_text,
                        'vector_weight': vector_weight,
                        'metrics': metrics
                    })
                    
                    # 打印详细比较
                    self.print_detailed_comparison(
                        f"{query_text} (权重:{vector_weight})", 
                        es_results, pd_results, metrics
                    )
                elif es_results:
                    print(f"    ⚠️ 只有ES返回了结果 ({len(es_results)}条)")
                elif pd_results:
                    print(f"    ⚠️ 只有ParadeDB返回了结果 ({len(pd_results)}条)")
                else:
                    print(f"    ❌ 两个引擎都没有返回结果")
        
        # 生成总结报告
        self.generate_summary_report(all_metrics)
    
    def generate_summary_report(self, all_metrics: List[Dict]):
        """生成总结报告"""
        if not all_metrics:
            print("\n❌ 没有可用的比较数据")
            return
        
        print(f"\n{'='*80}")
        print(f"📈 总结报告")
        print(f"{'='*80}")
        
        # 计算平均指标
        avg_rank_corr = np.mean([m['metrics'].rank_correlation for m in all_metrics])
        avg_score_corr = np.mean([m['metrics'].score_correlation for m in all_metrics])
        avg_score_diff = np.mean([m['metrics'].avg_score_diff for m in all_metrics])
        avg_common_docs = np.mean([m['metrics'].common_docs for m in all_metrics])
        
        print(f"\n🎯 平均指标:")
        print(f"  平均排名相关性: {avg_rank_corr:.3f}")
        print(f"  平均分数相关性: {avg_score_corr:.3f}")
        print(f"  平均分数差异: {avg_score_diff:.3f}")
        print(f"  平均共同文档数: {avg_common_docs:.1f}")
        
        # 按向量权重分组分析
        weight_groups = defaultdict(list)
        for m in all_metrics:
            weight_groups[m['vector_weight']].append(m['metrics'])
        
        print(f"\n📊 按向量权重分析:")
        for weight, metrics_list in weight_groups.items():
            avg_rank = np.mean([m.rank_correlation for m in metrics_list])
            avg_top5 = np.mean([m.top_k_overlap.get(5, 0) for m in metrics_list])
            print(f"  权重 {weight}: 排名相关性={avg_rank:.3f}, Top-5重叠率={avg_top5*100:.1f}%")
        
        # 一致性评估
        print(f"\n✅ 一致性评估:")
        if avg_rank_corr > 0.7:
            print(f"  🟢 排名一致性: 优秀 ({avg_rank_corr:.3f})")
        elif avg_rank_corr > 0.5:
            print(f"  🟡 排名一致性: 良好 ({avg_rank_corr:.3f})")
        else:
            print(f"  🔴 排名一致性: 需要改进 ({avg_rank_corr:.3f})")
        
        if avg_score_corr > 0.7:
            print(f"  🟢 分数一致性: 优秀 ({avg_score_corr:.3f})")
        elif avg_score_corr > 0.5:
            print(f"  🟡 分数一致性: 良好 ({avg_score_corr:.3f})")
        else:
            print(f"  🔴 分数一致性: 需要改进 ({avg_score_corr:.3f})")

def main():
    """主函数"""
    print("🔍 ES与ParadeDB搜索结果对比验证工具")
    print("=" * 50)
    
    # 验证配置
    if CONFIG:
        errors = validate_config()
        if errors:
            print("❌ 配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return
        print("✅ 配置验证通过")
    
    # 初始化比较器
    try:
        comparator = SearchEngineComparator()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return
    
    # 获取测试参数 - 优先级：自动生成配置 > 手动配置 > 默认配置
    try:
        from test_kb_config import COMPARISON_CONFIG
        index_name = COMPARISON_CONFIG['index_name']
        kb_ids = COMPARISON_CONFIG['kb_ids']
        test_queries = COMPARISON_CONFIG['test_queries']
        vector_weights = COMPARISON_CONFIG['vector_weights']
        print(f"📋 使用自动生成的测试配置")
    except ImportError:
        if CONFIG and 'test' in CONFIG:
            test_config = CONFIG['test']
            index_name = test_config['index_name']
            kb_ids = test_config['kb_ids']
            test_queries = test_config['test_queries']
            vector_weights = test_config['vector_weights']
            print(f"📋 使用手动配置文件中的参数")
        else:
            # 默认配置
            print("⚠️ 使用默认测试参数")
            print("💡 建议先运行 python simple_kb_setup.py 创建测试数据")
            index_name = "ragflow_6a2a5a8c00a611f0883a0242ac140006"
            kb_ids = ["6a2a5a8c-00a6-11f0-883a-0242ac140006"]
            test_queries = [
                "机器人", "人工智能", "自然语言处理", "深度学习算法", "数据分析方法",
                "zendesk", "customer service", "技术文档", "用户指南", "API接口"
            ]
            vector_weights = [0.3, 0.5, 0.7]
    
    print(f"  索引名称: {index_name}")
    print(f"  知识库ID: {kb_ids}")
    print(f"  测试查询数: {len(test_queries)}")
    print(f"  向量权重: {vector_weights}")
    
    # 运行比较测试
    try:
        comparator.run_comparison_test(
            index_name=index_name,
            kb_ids=kb_ids,
            test_queries=test_queries,
            vector_weights=vector_weights
        )
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 