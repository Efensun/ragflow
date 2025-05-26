#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2.extras import RealDictCursor
import json

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "rag_flow",
    "user": "postgres", 
    "password": "infini_rag_flow"
}

def check_search_fields():
    """检查搜索相关字段的数据情况"""
    
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 检查表中的搜索相关字段
            search_fields = [
                'id', 'content_with_weight', 'content_ltks', 'content_sm_ltks',
                'title_tks', 'title_sm_tks', 'important_kwd', 'important_tks',
                'question_tks', 'question_kwd'
            ]
            
            fields_str = ', '.join(search_fields)
            
            cur.execute(f"""
                SELECT {fields_str}
                FROM ragflow_6a2a5a8c00a611f0883a0242ac140006
                LIMIT 3
            """)
            
            results = cur.fetchall()
            
            print("=== 搜索字段数据检查 ===")
            for i, row in enumerate(results, 1):
                print(f"\n--- 记录 {i} (ID: {row['id']}) ---")
                for field in search_fields:
                    value = row.get(field)
                    if value is None:
                        print(f"{field}: NULL")
                    elif isinstance(value, str):
                        if len(value) > 100:
                            print(f"{field}: '{value[:100]}...' (长度: {len(value)})")
                        else:
                            print(f"{field}: '{value}' (长度: {len(value)})")
                    else:
                        print(f"{field}: {value} (类型: {type(value)})")
            
            # 检查是否有content_ltks字段包含"机器人"
            print("\n=== 检查content_ltks字段是否包含'机器人' ===")
            cur.execute("""
                SELECT id, content_ltks
                FROM ragflow_6a2a5a8c00a611f0883a0242ac140006
                WHERE content_ltks LIKE '%机器人%'
            """)
            
            ltks_results = cur.fetchall()
            print(f"包含'机器人'的content_ltks记录数: {len(ltks_results)}")
            
            for row in ltks_results:
                print(f"ID: {row['id']}")
                print(f"content_ltks: {row['content_ltks'][:200]}...")
            
            # 测试ParadeDB的全文搜索语法
            print("\n=== 测试ParadeDB全文搜索 ===")
            test_queries = [
                "content_ltks @@@ '机器人'",
                "content_with_weight @@@ '机器人'", 
                "title_tks @@@ '机器人'",
                "content_ltks LIKE '%机器人%'",
                "content_with_weight LIKE '%机器人%'"
            ]
            
            for query in test_queries:
                try:
                    cur.execute(f"""
                        SELECT COUNT(*) as count
                        FROM ragflow_6a2a5a8c00a611f0883a0242ac140006
                        WHERE {query}
                    """)
                    count = cur.fetchone()['count']
                    print(f"{query}: {count} 条记录")
                except Exception as e:
                    print(f"{query}: 错误 - {str(e)}")
                    
    except Exception as e:
        print(f"数据库查询错误: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_search_fields() 