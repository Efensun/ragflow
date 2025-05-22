#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from rag import settings
from rag.utils import singleton
from rag.utils.doc_store_conn import DocStoreConnection, MatchExpr, OrderByExpr, MatchTextExpr, MatchDenseExpr, \
    FusionExpr

ATTEMPT_TIME = 2
logger = logging.getLogger('ragflow.pg_conn')


@singleton
class PGConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        logger.info(f"Use ParadeDB {settings.PG['host']} as the doc engine.")

        for _ in range(ATTEMPT_TIME):
            try:
                # 创建连接池
                self.pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=settings.PG["host"],
                    port=settings.PG["port"],
                    database=settings.PG["database"],
                    user=settings.PG["user"],
                    password=settings.PG["password"]
                )

                # 测试连接
                with self.pool.getconn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT version()")
                        self.info["version"] = cur.fetchone()[0]
                    self.pool.putconn(conn)
                break
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting ParadeDB {settings.PG['host']} to be healthy.")
                time.sleep(5)

        if not self.info:
            msg = f"ParadeDB {settings.PG['host']} is unhealthy in 120s."
            logger.error(msg)
            raise Exception(msg)

        logger.info(f"ParadeDB {settings.PG['host']} is healthy.")

    """
    Database operations
    """

    def dbType(self) -> str:
        return "paradedb"

    def health(self) -> dict:
        with self.pool.getconn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                health_dict = dict(cur.fetchone())
                health_dict["type"] = "paradedb"
                return health_dict

    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    # 创建更完善的基础表
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {indexName} (
                            id TEXT PRIMARY KEY,
                            kb_id TEXT NOT NULL,
                            content TEXT,
                            title TEXT,
                            metadata JSONB,
                            available_int INTEGER DEFAULT 1,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            embedding vector({vectorSize})
                        )
                    """)
                    
                    # 创建触发器自动更新 updated_at 字段
                    cur.execute(f"""
                        CREATE OR REPLACE FUNCTION update_modified_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = CURRENT_TIMESTAMP;
                            RETURN NEW;
                        END;
                        $$ language 'plpgsql';
                    """)
                    
                    cur.execute(f"""
                        DROP TRIGGER IF EXISTS update_{indexName}_timestamp ON {indexName};
                        CREATE TRIGGER update_{indexName}_timestamp
                        BEFORE UPDATE ON {indexName}
                        FOR EACH ROW EXECUTE FUNCTION update_modified_column();
                    """)
                    
                    # 创建 kb_id 索引 - 提高按知识库过滤性能
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {indexName}_kb_id_idx ON {indexName} (kb_id);
                    """)
                    
                    # 创建BM25索引 - 用于全文搜索
                    cur.execute(f"""
                        CALL paradedb.create_bm25(
                            index_name => '{indexName}_bm25',
                            table_name => '{indexName}',
                            key_field => 'id',
                            text_fields => ARRAY[
                                paradedb.field('content'),
                                paradedb.field('title')
                            ],
                            numeric_fields => paradedb.field('kb_id'),
                            boolean_fields => paradedb.field('available_int', as_int => true),
                            json_fields => paradedb.field('metadata')
                        )
                    """)
                    
                    # 创建向量索引 - 用于向量相似度搜索
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {indexName}_vector_idx 
                        ON {indexName} 
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (
                            m = 16,               -- 每个节点的最大连接数
                            ef_construction = 64  -- 构建时的搜索宽度，越大越精确但也越慢
                        )
                    """)
                    
                    conn.commit()
                    return True
        except Exception as e:
            logger.exception(f"PGConnection.createIdx error {indexName}: {str(e)}")
            return False

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    # 删除BM25索引
                    cur.execute(f"DROP INDEX IF EXISTS {indexName}_bm25")
                    # 删除向量索引
                    cur.execute(f"DROP INDEX IF EXISTS {indexName}_vector_idx")
                    # 删除表
                    cur.execute(f"DROP TABLE IF EXISTS {indexName}")
                    conn.commit()
        except Exception:
            logger.exception(f"PGConnection.deleteIdx error {indexName}")

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (indexName,))
                    return cur.fetchone()[0]
        except Exception:
            logger.exception("PGConnection.indexExist got exception")
            return False

    """
    CRUD operations
    """

    def search(
            self, selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None
    ):
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")

        # 构建SELECT子句
        select_clause = []
        for field in selectFields:
            select_clause.append(field)

        # 添加高亮
        for field in highlightFields:
            select_clause.append(f"paradedb.snippet({field}) as {field}_hl")

        # 构建WHERE子句
        where_conditions = []
        params = []
        if knowledgebaseIds:
            where_conditions.append(f"kb_id = ANY(%s)")
            params.append(knowledgebaseIds)

        # 处理匹配表达式
        for expr in matchExprs:
            if isinstance(expr, MatchTextExpr):
                where_conditions.append(f"{expr.fields[0]} @@@ %s")
                params.append(expr.matching_text)
            elif isinstance(expr, MatchDenseExpr):
                where_conditions.append(f"embedding <=> %s <= %s")
                params.extend([expr.embedding_data, expr.topn])

        # 构建WHERE子句字符串
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        # 构建ORDER BY子句
        order_clause = []
        if orderBy and orderBy.fields:
            for field, order in orderBy.fields:
                order_clause.append(f"{field} {'DESC' if order == 1 else 'ASC'}")

        # 构建完整SQL
        sql = f"""
            SELECT {', '.join(select_clause)}
            FROM {indexNames[0]}
            {where_clause}
            {"ORDER BY " + ", ".join(order_clause) if order_clause else ""}
            {"LIMIT " + str(limit) if limit > 0 else ""}
            {"OFFSET " + str(offset) if offset > 0 else ""}
        """

        try:
            with self.pool.getconn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # 执行主查询
                    cur.execute(sql, params)
                    results = {
                        "hits": {
                            "total": cur.rowcount,
                            "hits": cur.fetchall()
                        }
                    }

                    # 如果需要聚合字段，执行聚合查询
                    if aggFields:
                        results["aggregations"] = {}
                        
                        for field in aggFields:
                            # 构建聚合查询
                            agg_sql = f"""
                                SELECT {field}, COUNT(*) as doc_count
                                FROM {indexNames[0]}
                                {where_clause}
                                GROUP BY {field}
                                ORDER BY doc_count DESC
                            """
                            
                            # 执行聚合查询
                            cur.execute(agg_sql, params)
                            
                            # 构建与ES兼容的聚合结果格式
                            buckets = []
                            for row in cur.fetchall():
                                if row[field] is not None:  # 忽略NULL值
                                    buckets.append({
                                        "key": row[field],
                                        "doc_count": row["doc_count"]
                                    })
                            
                            # 将聚合结果添加到响应中
                            results["aggregations"][f"aggs_{field}"] = {
                                "buckets": buckets
                            }
                    
                    return results
        except Exception as e:
            logger.exception(f"PGConnection.search error: {str(e)}")
            raise e

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        try:
            with self.pool.getconn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT * FROM {indexName} 
                        WHERE id = %s AND kb_id = ANY(%s)
                    """, (chunkId, knowledgebaseIds))
                    result = cur.fetchone()
                    if result:
                        return dict(result)
                    return None
        except Exception:
            logger.exception(f"PGConnection.get({chunkId}) got exception")
            return None

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        errors = []
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    for doc in documents:
                        try:
                            doc_id = doc.pop("id", "")
                            doc["kb_id"] = knowledgebaseId

                            fields = ", ".join(doc.keys())
                            placeholders = ", ".join(["%s"] * len(doc))

                            sql = f"""
                                INSERT INTO {indexName} (id, {fields})
                                VALUES (%s, {placeholders})
                                ON CONFLICT (id) DO UPDATE
                                SET ({fields}) = ({placeholders})
                            """

                            cur.execute(sql, [doc_id] + list(doc.values()) * 2)
                        except Exception as e:
                            errors.append(f"{doc_id}:{str(e)}")

                    conn.commit()
            return errors
        except Exception as e:
            logger.exception("PGConnection.insert got exception")
            return [str(e)]

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    # 构建SET子句
                    set_clause = []
                    params = []
                    for k, v in newValue.items():
                        set_clause.append(f"{k} = %s")
                        params.append(v)

                    # 构建WHERE子句
                    where_conditions = []
                    if knowledgebaseId:
                        where_conditions.append("kb_id = %s")
                        params.append(knowledgebaseId)

                    if "id" in condition:
                        where_conditions.append("id = %s")
                        params.append(condition["id"])

                    sql = f"""
                        UPDATE {indexName}
                        SET {', '.join(set_clause)}
                        WHERE {' AND '.join(where_conditions)}
                    """

                    cur.execute(sql, params)
                    conn.commit()
                    return True
        except Exception:
            logger.exception("PGConnection.update got exception")
            return False

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    where_conditions = ["kb_id = %s"]
                    params = [knowledgebaseId]

                    if "id" in condition:
                        chunk_ids = condition["id"]
                        if not isinstance(chunk_ids, list):
                            chunk_ids = [chunk_ids]
                        if chunk_ids:
                            where_conditions.append("id = ANY(%s)")
                            params.append(chunk_ids)

                    sql = f"""
                        DELETE FROM {indexName}
                        WHERE {' AND '.join(where_conditions)}
                        RETURNING id
                    """

                    cur.execute(sql, params)
                    deleted = cur.rowcount
                    conn.commit()
                    return deleted
        except Exception:
            logger.exception("PGConnection.delete got exception")
            return 0

    """
    Helper functions for search result
    """

    def getTotal(self, res):
        return res["hits"]["total"]

    def getChunkIds(self, res):
        return [hit["id"] for hit in res["hits"]["hits"]]

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        result = {}
        for hit in res["hits"]["hits"]:
            doc_id = hit["id"]
            field_values = {}
            for field in fields:
                if field in hit:
                    field_values[field] = hit[field]
            if field_values:
                result[doc_id] = field_values
        return result

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        highlights = {}
        for hit in res["hits"]["hits"]:
            doc_id = hit["id"]
            hl_field = f"{fieldnm}_hl"
            if hl_field in hit:
                highlights[doc_id] = hit[hl_field]
        return highlights

    def getAggregation(self, res, fieldnm: str):
        """
        从搜索结果中提取特定字段的聚合结果
        """
        agg_field = "aggs_" + fieldnm
        if "aggregations" not in res or agg_field not in res["aggregations"]:
            return list()
        
        buckets = res["aggregations"][agg_field]["buckets"]
        return [(b["key"], b["doc_count"]) for b in buckets]

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.closeall()