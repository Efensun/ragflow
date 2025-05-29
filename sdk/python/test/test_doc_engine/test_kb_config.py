# 自动生成的搜索对比测试配置（真实向量）
# 生成时间: 2025-05-28 23:29:29
# Embedding类型: 真实向量

INDEX_NAME = "ragflow_3c12941dc2ba4416865ab4d1280e1ddf"
KB_ID = "2bc452d7-3bd8-406b-b47c-76f6ab50cc39"
TENANT_ID = "3c12941d-c2ba-4416-865a-b4d1280e1ddf"

# 用于 quick_search_test.py
TEST_CONFIG = {
    "index_name": "ragflow_3c12941dc2ba4416865ab4d1280e1ddf",
    "kb_ids": ["2bc452d7-3bd8-406b-b47c-76f6ab50cc39"],
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
}

# 用于 compare_search_engines.py
COMPARISON_CONFIG = {
    "index_name": "ragflow_3c12941dc2ba4416865ab4d1280e1ddf",
    "kb_ids": ["2bc452d7-3bd8-406b-b47c-76f6ab50cc39"],
    "vector_weights": [0.2, 0.3, 0.5, 0.7, 0.8],  # 更多权重测试
    "test_queries": TEST_CONFIG["test_queries"]
}
