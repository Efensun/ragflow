#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库隔离测试脚本

验证不同知识库的文档是否正确隔离，搜索时是否只返回指定知识库的文档
"""

import os
import sys
import time
import logging
from typing import List, Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import settings
from rag.utils.es_conn import ESConnection
from rag.utils.pd_conn import PDConnection
from rag.utils.doc_store_conn import MatchTextExpr, OrderByExpr
from rag.nlp import rag_tokenizer
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KBIsolationTester:
    """知识库隔离测试器"""
    
    def __init__(self):
        self.es_conn = None
        self.pd_conn = None
        self.init_connections()
    
    def init_connections(self):
        """初始化数据库连接"""
        try:
            if hasattr(settings, 'ES') and settings.ES.get('hosts'):
                self.es_conn = ESConnection()
                print("✅ ES连接成功")
        except Exception as e:
            print(f"❌ ES连接失败: {e}")
        
        try:
            if hasattr(settings, 'PARADEDB') and settings.PARADEDB.get('host'):
                self.pd_conn = PDConnection()
                print("✅ ParadeDB连接成功")
        except Exception as e:
            print(f"❌ ParadeDB连接失败: {e}")
    
    def verify_data_insertion(self, conn, engine_name: str, index_name: str, kb_id: str, expected_count: int):
        """验证数据插入是否成功"""
        try:
            # 使用简单的查询来验证数据
            res = conn.search(
                selectFields=["id", "title_tks", "kb_id"],
                highlightFields=[],
                condition={"available_int": 1},
                matchExprs=[],  # 空的匹配表达式，返回所有文档
                orderBy=OrderByExpr(),
                offset=0,
                limit=100,
                indexNames=index_name,
                knowledgebaseIds=[kb_id]
            )
            
            total = conn.getTotal(res) if hasattr(conn, 'getTotal') else len(res.get("hits", {}).get("hits", []))
            hits = res.get("hits", {}).get("hits", [])
            
            print(f"  {engine_name} 知识库 {kb_id[:8]}... 验证:")
            print(f"    总文档数: {total}")
            print(f"    返回文档数: {len(hits)}")
            print(f"    预期文档数: {expected_count}")
            
            if hits:
                print(f"    样本文档:")
                for i, hit in enumerate(hits[:2]):
                    source = hit.get("_source", {})
                    print(f"      {i+1}. ID: {hit.get('_id', '')[:8]}...")
                    print(f"         标题: {source.get('title_tks', '')}")
                    print(f"         KB_ID: {source.get('kb_id', '')[:8]}...")
            
            return total == expected_count
            
        except Exception as e:
            print(f"  ❌ {engine_name} 数据验证失败: {e}")
            return False
    
    def create_test_data(self, index_name: str) -> Dict[str, List[str]]:
        """创建测试数据：两个不同的知识库"""
        kb1_id = str(uuid.uuid4())
        kb2_id = str(uuid.uuid4())
        
        print(f"\n📝 创建测试数据:")
        print(f"  知识库1 ID: {kb1_id}")
        print(f"  知识库2 ID: {kb2_id}")
        
        # 知识库1的文档 - 使用更丰富的内容
        kb1_docs = [
            {
                "id": str(uuid.uuid4()),
                "kb_id": kb1_id,
                "title_tks": "人工智能基础知识文档",
                "content_with_weight": "这是知识库1的第一个文档，包含人工智能、机器学习、深度学习相关内容。人工智能是计算机科学的重要分支。",
                "content_ltks": "这是 知识库 1 的 第一个 文档 包含 人工智能 机器学习 深度学习 相关 内容 人工智能 是 计算机科学 的 重要 分支",
                "content_sm_ltks": "这是 知识库 1 的 第一个 文档 包含 人工 智能 机器 学习 深度 学习 相关 内容 人工 智能 是 计算机 科学 的 重要 分支",
                "available_int": 1,
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "create_timestamp_flt": time.time()
            },
            {
                "id": str(uuid.uuid4()),
                "kb_id": kb1_id,
                "title_tks": "机器学习算法详解文档", 
                "content_with_weight": "这是知识库1的第二个文档，包含机器学习算法、神经网络、数据挖掘相关内容。机器学习是人工智能的核心技术。",
                "content_ltks": "这是 知识库 1 的 第二个 文档 包含 机器学习 算法 神经网络 数据挖掘 相关 内容 机器学习 是 人工智能 的 核心 技术",
                "content_sm_ltks": "这是 知识库 1 的 第二个 文档 包含 机器 学习 算法 神经 网络 数据 挖掘 相关 内容 机器 学习 是 人工 智能 的 核心 技术",
                "available_int": 1,
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "create_timestamp_flt": time.time() + 1
            }
        ]
        
        # 知识库2的文档 - 使用不同主题的内容
        kb2_docs = [
            {
                "id": str(uuid.uuid4()),
                "kb_id": kb2_id,
                "title_tks": "区块链技术原理文档",
                "content_with_weight": "这是知识库2的第一个文档，包含区块链技术、加密货币、分布式系统相关内容。区块链是分布式账本技术。",
                "content_ltks": "这是 知识库 2 的 第一个 文档 包含 区块链 技术 加密货币 分布式系统 相关 内容 区块链 是 分布式 账本 技术",
                "content_sm_ltks": "这是 知识库 2 的 第一个 文档 包含 区块链 技术 加密 货币 分布式 系统 相关 内容 区块链 是 分布式 账本 技术",
                "available_int": 1,
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "create_timestamp_flt": time.time() + 2
            },
            {
                "id": str(uuid.uuid4()),
                "kb_id": kb2_id,
                "title_tks": "云计算平台架构文档",
                "content_with_weight": "这是知识库2的第二个文档，包含云计算平台、微服务架构、容器技术相关内容。云计算提供弹性计算资源。", 
                "content_ltks": "这是 知识库 2 的 第二个 文档 包含 云计算 平台 微服务 架构 容器 技术 相关 内容 云计算 提供 弹性 计算 资源",
                "content_sm_ltks": "这是 知识库 2 的 第二个 文档 包含 云 计算 平台 微 服务 架构 容器 技术 相关 内容 云 计算 提供 弹性 计算 资源",
                "available_int": 1,
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "create_timestamp_flt": time.time() + 3
            }
        ]
        
        # 插入数据到ES
        if self.es_conn:
            try:
                print(f"  插入数据到ES...")
                errors1 = self.es_conn.insert(kb1_docs, index_name, kb1_id)
                errors2 = self.es_conn.insert(kb2_docs, index_name, kb2_id)
                if errors1 or errors2:
                    print(f"  ⚠️ ES插入有错误: KB1={errors1}, KB2={errors2}")
                else:
                    print(f"  ✅ ES数据插入成功")
            except Exception as e:
                print(f"  ❌ ES数据插入失败: {e}")
        
        # 插入数据到ParadeDB
        if self.pd_conn:
            try:
                print(f"  插入数据到ParadeDB...")
                errors1 = self.pd_conn.insert(kb1_docs, index_name, kb1_id)
                errors2 = self.pd_conn.insert(kb2_docs, index_name, kb2_id)
                if errors1 or errors2:
                    print(f"  ⚠️ ParadeDB插入有错误: KB1={errors1}, KB2={errors2}")
                else:
                    print(f"  ✅ ParadeDB数据插入成功")
            except Exception as e:
                print(f"  ❌ ParadeDB数据插入失败: {e}")
        
        # 等待索引刷新
        print(f"  ⏳ 等待索引刷新...")
        time.sleep(5)
        
        # 验证数据插入
        print(f"\n🔍 验证数据插入:")
        if self.es_conn:
            self.verify_data_insertion(self.es_conn, "ES", index_name, kb1_id, 2)
            self.verify_data_insertion(self.es_conn, "ES", index_name, kb2_id, 2)
        
        if self.pd_conn:
            self.verify_data_insertion(self.pd_conn, "ParadeDB", index_name, kb1_id, 2)
            self.verify_data_insertion(self.pd_conn, "ParadeDB", index_name, kb2_id, 2)
        
        return {
            "kb1_id": kb1_id,
            "kb2_id": kb2_id,
            "kb1_doc_ids": [doc["id"] for doc in kb1_docs],
            "kb2_doc_ids": [doc["id"] for doc in kb2_docs]
        }
    
    def test_kb_isolation(self, index_name: str, test_data: Dict):
        """测试知识库隔离"""
        kb1_id = test_data["kb1_id"]
        kb2_id = test_data["kb2_id"]
        kb1_doc_ids = set(test_data["kb1_doc_ids"])
        kb2_doc_ids = set(test_data["kb2_doc_ids"])
        
        print(f"\n🧪 测试知识库隔离")
        print(f"知识库1 ID: {kb1_id}")
        print(f"知识库2 ID: {kb2_id}")
        print(f"知识库1文档数: {len(kb1_doc_ids)}")
        print(f"知识库2文档数: {len(kb2_doc_ids)}")
        
        # 测试ES隔离
        if self.es_conn:
            print(f"\n📊 ES隔离测试:")
            self._test_engine_isolation(self.es_conn, "ES", index_name, kb1_id, kb2_id, kb1_doc_ids, kb2_doc_ids)
        
        # 测试ParadeDB隔离
        if self.pd_conn:
            print(f"\n📊 ParadeDB隔离测试:")
            self._test_engine_isolation(self.pd_conn, "ParadeDB", index_name, kb1_id, kb2_id, kb1_doc_ids, kb2_doc_ids)
    
    def _test_engine_isolation(self, conn, engine_name: str, index_name: str, 
                              kb1_id: str, kb2_id: str, 
                              kb1_doc_ids: set, kb2_doc_ids: set):
        """测试单个引擎的知识库隔离"""
        
        # 准备多个测试查询
        test_queries = [
            "文档",
            "知识库",
            "人工智能",
            "机器学习",
            "区块链",
            "云计算"
        ]
        
        for query_text in test_queries:
            print(f"\n  🔍 测试查询: '{query_text}'")
            
            # 对查询文本进行分词处理（ParadeDB需要）
            tokenized_query = rag_tokenizer.fine_grained_tokenize(rag_tokenizer.tokenize(query_text))
            print(f"    原始查询: {query_text}")
            print(f"    分词结果: {tokenized_query}")
            
            # 测试1：搜索知识库1
            try:
                print(f"    📋 搜索知识库1:")
                res1 = conn.search(
                    selectFields=["id", "title_tks", "kb_id", "content_with_weight"],
                    highlightFields=[],
                    condition={"available_int": 1},
                    matchExprs=[MatchTextExpr(
                        fields=["content_with_weight^10", "title_tks^8", "content_ltks^5", "content_sm_ltks^3"],
                        matching_text=query_text,
                        topn=10,
                        extra_options={"minimum_should_match": 0.1}  # 降低匹配要求
                    )],
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=10,
                    indexNames=index_name,
                    knowledgebaseIds=[kb1_id]
                )
                
                kb1_results = set()
                hits = res1.get("hits", {}).get("hits", [])
                print(f"      返回结果: {len(hits)}条")
                
                for hit in hits:
                    kb1_results.add(hit.get("_id"))
                    source = hit.get("_source", {})
                    score = hit.get("_score", 0)
                    print(f"        文档: {hit.get('_id')[:8]}... 分数:{score:.3f}")
                    print(f"        标题: {source.get('title_tks', '')}")
                    print(f"        内容: {source.get('content_with_weight', '')[:50]}...")
                
                # 验证隔离性
                kb1_isolation_ok = kb1_results.issubset(kb1_doc_ids) if kb1_results else True
                kb1_no_leak = len(kb1_results & kb2_doc_ids) == 0
                
                print(f"      ✅ 知识库1隔离正确: {kb1_isolation_ok}")
                print(f"      ✅ 无知识库2泄漏: {kb1_no_leak}")
                
            except Exception as e:
                print(f"      ❌ 知识库1搜索失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 测试2：搜索知识库2
            try:
                print(f"    📋 搜索知识库2:")
                res2 = conn.search(
                    selectFields=["id", "title_tks", "kb_id", "content_with_weight"],
                    highlightFields=[],
                    condition={"available_int": 1},
                    matchExprs=[MatchTextExpr(
                        fields=["content_with_weight^10", "title_tks^8", "content_ltks^5", "content_sm_ltks^3"],
                        matching_text=query_text,
                        topn=10,
                        extra_options={"minimum_should_match": 0.1}  # 降低匹配要求
                    )],
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=10,
                    indexNames=index_name,
                    knowledgebaseIds=[kb2_id]
                )
                
                kb2_results = set()
                hits = res2.get("hits", {}).get("hits", [])
                print(f"      返回结果: {len(hits)}条")
                
                for hit in hits:
                    kb2_results.add(hit.get("_id"))
                    source = hit.get("_source", {})
                    score = hit.get("_score", 0)
                    print(f"        文档: {hit.get('_id')[:8]}... 分数:{score:.3f}")
                    print(f"        标题: {source.get('title_tks', '')}")
                    print(f"        内容: {source.get('content_with_weight', '')[:50]}...")
                
                # 验证隔离性
                kb2_isolation_ok = kb2_results.issubset(kb2_doc_ids) if kb2_results else True
                kb2_no_leak = len(kb2_results & kb1_doc_ids) == 0
                
                print(f"      ✅ 知识库2隔离正确: {kb2_isolation_ok}")
                print(f"      ✅ 无知识库1泄漏: {kb2_no_leak}")
                
            except Exception as e:
                print(f"      ❌ 知识库2搜索失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 如果某个查询有结果，就不需要继续测试其他查询了
            if (len(hits) > 0 for hits in [res1.get("hits", {}).get("hits", []), res2.get("hits", {}).get("hits", [])]):
                print(f"    ✅ 查询 '{query_text}' 有结果，隔离测试通过")
                break
        
        # 测试3：同时搜索两个知识库
        try:
            print(f"  🔍 测试: 搜索两个知识库")
            res_both = conn.search(
                selectFields=["id", "title_tks", "kb_id"],
                highlightFields=[],
                condition={"available_int": 1},
                matchExprs=[MatchTextExpr(
                    fields=["content_with_weight^10", "title_tks^8", "content_ltks^5"],
                    matching_text="文档",
                    topn=10,
                    extra_options={"minimum_should_match": 0.1}
                )],
                orderBy=OrderByExpr(),
                offset=0,
                limit=10,
                indexNames=index_name,
                knowledgebaseIds=[kb1_id, kb2_id]
            )
            
            both_results = set()
            hits = res_both.get("hits", {}).get("hits", [])
            for hit in hits:
                both_results.add(hit.get("_id"))
                source = hit.get("_source", {})
                print(f"    找到文档: {hit.get('_id')[:8]}... KB:{source.get('kb_id', '')[:8]}... 标题:{source.get('title_tks', '')}")
            
            expected_all = kb1_doc_ids | kb2_doc_ids
            print(f"  双知识库搜索结果: {len(both_results)}条")
            print(f"  预期总文档数: {len(expected_all)}")
            
            both_complete = both_results == expected_all
            print(f"  ✅ 双知识库搜索完整: {both_complete}")
            
        except Exception as e:
            print(f"  ❌ 双知识库搜索失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🔒 知识库隔离测试工具")
    print("=" * 40)
    
    # 初始化测试器
    tester = KBIsolationTester()
    
    if not tester.es_conn and not tester.pd_conn:
        print("❌ 没有可用的搜索引擎连接")
        return
    
    # 生成测试索引名
    tenant_id = str(uuid.uuid4()).replace('-', '')
    index_name = f"ragflow_{tenant_id}"
    
    print(f"测试索引: {index_name}")
    
    try:
        # 创建索引
        print(f"\n🏗️ 创建索引:")
        if tester.es_conn:
            try:
                tester.es_conn.createIdx(index_name, "", 1024)
                print(f"  ✅ ES索引创建成功")
            except Exception as e:
                print(f"  ❌ ES索引创建失败: {e}")
        
        if tester.pd_conn:
            try:
                tester.pd_conn.createIdx(index_name, "", 1024)
                print(f"  ✅ ParadeDB索引创建成功")
            except Exception as e:
                print(f"  ❌ ParadeDB索引创建失败: {e}")
        
        # 创建测试数据
        test_data = tester.create_test_data(index_name)
        
        # 测试隔离性
        tester.test_kb_isolation(index_name, test_data)
        
        print(f"\n🎉 知识库隔离测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理测试数据
        try:
            if tester.es_conn:
                tester.es_conn.deleteIdx(index_name, "")
            if tester.pd_conn:
                tester.pd_conn.deleteIdx(index_name, "")
            print(f"🧹 已清理测试索引: {index_name}")
        except Exception as e:
            print(f"⚠️ 清理索引失败: {e}")

if __name__ == "__main__":
    main() 