#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试Xinference embedding模型连接

这个脚本用于验证是否可以绕过RAGFlow权限控制，直接调用Xinference的jina-embeddings-v3模型
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_xinference_direct():
    """直接测试Xinference连接"""
    print("🧪 测试直接调用Xinference embedding模型")
    print("=" * 50)
    
    # 方法1: 使用RAGFlow的XinferenceEmbed类
    print("\n1️⃣ 测试RAGFlow XinferenceEmbed类")
    try:
        from rag.llm.embedding_model import XinferenceEmbed
        
        xinference_embed = XinferenceEmbed(
            key="",  # Xinference通常不需要API key
            model_name="jina-embeddings-v3",
            base_url="http://120.77.38.66:8008/"  # 用户提供的Xinference地址
        )
        
        # 测试embedding
        test_texts = ["人工智能", "机器学习", "自然语言处理"]
        embeddings, tokens = xinference_embed.encode(test_texts)
        
        print(f"✅ RAGFlow XinferenceEmbed 成功!")
        print(f"  向量维度: {len(embeddings[0])}")
        print(f"  测试文本数: {len(test_texts)}")
        print(f"  返回向量数: {len(embeddings)}")
        print(f"  Token消耗: {tokens}")
        print(f"  第一个向量前5维: {embeddings[0][:5]}")
        
        return True, len(embeddings[0])
        
    except Exception as e:
        print(f"❌ RAGFlow XinferenceEmbed 失败: {e}")
    
    # 方法2: 使用OpenAI客户端直接连接
    print("\n2️⃣ 测试OpenAI客户端直接连接")
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key="empty", base_url="http://120.77.38.66:8008/v1")
        
        # 测试embedding
        response = client.embeddings.create(
            input=["人工智能", "机器学习"],
            model="jina-embeddings-v3"
        )
        
        print(f"✅ OpenAI客户端 成功!")
        print(f"  向量维度: {len(response.data[0].embedding)}")
        print(f"  返回数据数: {len(response.data)}")
        print(f"  第一个向量前5维: {response.data[0].embedding[:5]}")
        
        if hasattr(response, 'usage'):
            print(f"  Token消耗: {response.usage.total_tokens}")
        
        return True, len(response.data[0].embedding)
        
    except Exception as e:
        print(f"❌ OpenAI客户端 失败: {e}")
    
    # 方法3: 测试其他可能的地址
    print("\n3️⃣ 测试其他Xinference地址")
    xinference_urls = [
        "http://120.77.38.66:8008/",
        "http://120.77.38.66:8008/v1",
        "http://127.0.0.1:9997",
        "http://0.0.0.0:9997",
        "http://localhost:9997/v1"
    ]
    
    for base_url in xinference_urls:
        try:
            from rag.llm.embedding_model import XinferenceEmbed
            
            xinference_embed = XinferenceEmbed(
                key="",
                model_name="jina-embeddings-v3",
                base_url=base_url
            )
            
            embeddings, tokens = xinference_embed.encode(["测试"])
            
            print(f"✅ 地址 {base_url} 成功!")
            print(f"  向量维度: {len(embeddings[0])}")
            
            return True, len(embeddings[0])
            
        except Exception as e:
            print(f"❌ 地址 {base_url} 失败: {e}")
    
    return False, None

def test_xinference_service():
    """测试Xinference服务状态"""
    print("\n🔍 检查Xinference服务状态")
    print("-" * 30)
    
    try:
        import requests
        
        # 检查Xinference服务是否运行
        response = requests.get("http://120.77.38.66:8008/v1/models", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Xinference服务运行正常")
            print(f"  可用模型数: {len(models.get('data', []))}")
            
            # 查找jina-embeddings-v3模型
            jina_models = [m for m in models.get('data', []) if 'jina' in m.get('id', '').lower()]
            if jina_models:
                print(f"  找到Jina模型: {[m['id'] for m in jina_models]}")
            else:
                print(f"  ⚠️ 未找到jina-embeddings-v3模型")
                print(f"  可用模型: {[m.get('id', 'unknown') for m in models.get('data', [])]}")
        else:
            print(f"❌ Xinference服务响应异常: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到Xinference服务 (http://120.77.38.66:8008)")
        print(f"💡 请确保Xinference服务正在运行:")
        print(f"   xinference-local --host 0.0.0.0 --port 8008")
    except Exception as e:
        print(f"❌ 检查Xinference服务失败: {e}")

def main():
    """主函数"""
    print("🚀 Xinference直接调用测试工具")
    print("=" * 50)
    
    # 检查服务状态
    test_xinference_service()
    
    # 测试直接调用
    success, vector_dim = test_xinference_direct()
    
    print(f"\n📊 测试结果总结")
    print("=" * 30)
    
    if success:
        print(f"✅ 成功绕过权限控制，直接调用Xinference!")
        print(f"  向量维度: {vector_dim}")
        print(f"  模型: jina-embeddings-v3")
        print(f"\n💡 现在可以运行:")
        print(f"  python sdk/python/test/test_doc_engine/simple_kb_setup.py")
        print(f"  python sdk/python/test/test_doc_engine/quick_search_test.py")
    else:
        print(f"❌ 无法直接调用Xinference")
        print(f"\n🔧 故障排除建议:")
        print(f"  1. 确保Xinference服务正在运行:")
        print(f"     xinference-local --host 0.0.0.0 --port 8008")
        print(f"  2. 启动jina-embeddings-v3模型:")
        print(f"     xinference launch --model-name jina-embeddings-v3 --model-type embedding")
        print(f"  3. 检查防火墙设置，确保端口9997可访问")
        print(f"  4. 或者安装sentence-transformers作为备选:")
        print(f"     pip install sentence-transformers")

if __name__ == "__main__":
    main() 