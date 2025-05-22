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
import re

ATTEMPT_TIME = 2
logger = logging.getLogger('ragflow.pd_conn')


@singleton
class PDConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        logger.info(f"Use ParadeDB {settings.PARADEDB['host']} as the doc engine.")

        for _ in range(ATTEMPT_TIME):
            try:
                # 创建连接池
                self.pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=settings.PARADEDB["host"],
                    port=settings.PARADEDB["port"],
                    database=settings.PARADEDB.get("database", settings.PARADEDB.get("name", "rag_flow")),
                    user=settings.PARADEDB["user"],
                    password=settings.PARADEDB["password"]
                )

                # 测试连接
                with self.pool.getconn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT version()")
                        self.info["version"] = cur.fetchone()[0]
                    self.pool.putconn(conn)
                break
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting ParadeDB {settings.PARADEDB['host']} to be healthy.")
                time.sleep(5)

        if not self.info:
            msg = f"ParadeDB {settings.PARADEDB['host']} is unhealthy in 120s."
            logger.error(msg)
            raise Exception(msg)

        logger.info(f"ParadeDB {settings.PARADEDB['host']} is healthy.")

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
                        CREATE INDEX {indexName}_bm25 ON {indexName}
                        USING bm25 (id, content, title, kb_id, available_int, metadata)
                        WITH (key_field='id');
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
            logger.exception(f"PDConnection.createIdx error {indexName}: {str(e)}")
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
            logger.exception(f"PDConnection.deleteIdx error {indexName}")

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
            logger.exception("PDConnection.indexExist got exception")
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

        # 检查表是否存在，如果不存在则返回空结果
        if not self.indexExist(indexNames[0]):
            logger.warning(f"Table {indexNames[0]} does not exist, returning empty result")
            return {
                "hits": {
                    "total": 0,
                    "hits": []
                }
            }

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
        has_vector_search = False
        vector_query = None
        vector_topn = 10
        vector_similarity_weight = 0.5  # 默认权重
        has_text_search = False
        text_query = None

        # 首先检查是否有FusionExpr并提取权重
        for m in matchExprs:
            if isinstance(m, FusionExpr) and m.method == "weighted_sum" and "weights" in m.fusion_params:
                # 假设匹配表达式顺序是：[MatchTextExpr, MatchDenseExpr, FusionExpr]
                weights = m.fusion_params["weights"]
                # 解析权重字符串，格式为: "文本权重, 向量权重"
                weight_parts = weights.split(",")
                if len(weight_parts) >= 2:
                    try:
                        # 文本搜索权重
                        text_weight = float(weight_parts[0].strip())
                        # 向量搜索权重
                        vector_similarity_weight = float(weight_parts[1].strip())
                        logger.debug(f"Using weights: text={text_weight}, vector={vector_similarity_weight}")
                    except ValueError:
                        logger.warning(f"Invalid weight format: {weights}, using default weight: text=0.5, vector={vector_similarity_weight}")

        # 然后处理各种匹配表达式
        for expr in matchExprs:
            if isinstance(expr, MatchTextExpr):
                has_text_search = True
                text_query = expr.matching_text
                where_conditions.append(f"{expr.fields[0]} @@@ %s")
                params.append(expr.matching_text)
            elif isinstance(expr, MatchDenseExpr):
                # 标记存在向量搜索，稍后在ORDER BY中处理
                has_vector_search = True
                vector_query = expr.embedding_data
                vector_topn = expr.topn

        # 构建WHERE子句字符串
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        # 构建ORDER BY子句
        order_clause = []

        # 混合搜索排序策略
        # 与ES连接保持一致：无论权重如何都同时考虑文本搜索和向量搜索
        # 参考ES连接实现方式: bqry.boost = 1.0 - vector_similarity_weight
        if has_vector_search and has_text_search:
            # 计算文本搜索权重
            text_weight = 1.0 - vector_similarity_weight
            
            # 在WHERE子句中已经添加了文本搜索条件
            # 在ORDER BY中添加向量搜索排序，确保两种搜索方式都被使用
            order_clause = [f"embedding <=> %s"]
            params.append(vector_query)
            logger.info(f"Using hybrid search with weights: text={text_weight}, vector={vector_similarity_weight}")
        elif has_vector_search:
            # 只有向量搜索
            order_clause = [f"embedding <=> %s"]
            params.append(vector_query)
            logger.info("Using vector search only")
        elif has_text_search:
            # 只有文本搜索，ParadeDB默认使用BM25排序
            logger.info("Using text search only with BM25 ranking")

        # 如果还有其他排序条件，添加到后面
        if orderBy and orderBy.fields:
            for field, order in orderBy.fields:
                order_clause.append(f"{field} {'DESC' if order == 1 else 'ASC'}")
        
        # 构建完整SQL
        sql = f"""
            SELECT {', '.join(select_clause)}
            FROM {indexNames[0]}
            {where_clause}
            {"ORDER BY " + ", ".join(order_clause) if order_clause else ""}
            {"LIMIT " + str(vector_topn) if has_vector_search else str(limit) if limit > 0 else ""}
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
        except psycopg2.errors.UndefinedTable as e:
            # 表不存在的情况，返回空结果而不是抛出异常
            logger.warning(f"Table {indexNames[0]} does not exist: {str(e)}")
            return {
                "hits": {
                    "total": 0,
                    "hits": []
                }
            }
        except Exception as e:
            logger.exception(f"PDConnection.search error: {str(e)}")
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
            logger.exception(f"PDConnection.get({chunkId}) got exception")
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
            logger.exception("PDConnection.insert got exception")
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
            logger.exception("PDConnection.update got exception")
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
            logger.exception("PDConnection.delete got exception")
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

    """
    SQL 执行方法
    """
    def sql(self, sql: str, fetch_size: int, format: str):
        """
        执行SQL查询并返回结果
        """
        logger.debug(f"PDConnection.sql get sql: {sql}")
        
        # SQL预处理 - 类似ES实现的规范化
        sql = re.sub(r"[ `]+", " ", sql)
        sql = sql.replace("%", "")
        
        # 处理全文搜索优化 - 将LIKE转换为ParadeDB的全文搜索语法
        # 示例: content_ltks LIKE '%word%' => content_ltks @@@ 'word'
        replaces = []
        for r in re.finditer(r" ([a-z_]+_l?tks)( like | ?= ?)'([^']+)'", sql):
            fld, v = r.group(1), r.group(3)
            # 使用ParadeDB的BM25全文搜索
            match = " {} @@@ '{}' ".format(fld, v.replace('%', '').strip())
            replaces.append(
                ("{}{}'{}'".format(
                    r.group(1),
                    r.group(2),
                    r.group(3)),
                 match))

        for p, r in replaces:
            sql = sql.replace(p, r, 1)
            
        logger.debug(f"PDConnection.sql to paradedb: {sql}")
        
        # 添加重试机制
        for i in range(ATTEMPT_TIME):
            try:
                with self.pool.getconn() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(sql)
                        
                        if format == "json":
                            # 获取所有列名
                            columns = [{"name": desc[0], "type": "text"} for desc in cur.description]
                            
                            # 获取结果行
                            rows = cur.fetchmany(fetch_size)
                            
                            # 构建与ES兼容的响应格式
                            result = {
                                "columns": columns,
                                "rows": [[row[col["name"]] for col in columns] for row in rows],
                                "cursor": None  # ParadeDB不支持游标，设为None
                            }
                            
                            return result
                        else:
                            # 返回默认格式
                            return {"results": [dict(row) for row in cur.fetchmany(fetch_size)]}
            except psycopg2.OperationalError as e:
                # 连接超时处理
                logger.warning(f"PDConnection.sql connection timeout: {str(e)}")
                if i < ATTEMPT_TIME - 1:  # 如果不是最后一次尝试
                    time.sleep(3)  # 等待一段时间再重试
                    continue
                return {"error": f"Connection error: {str(e)}"}
            except Exception as e:
                logger.exception(f"PDConnection.sql error: {str(e)}")
                # 提供详细错误信息
                error_msg = str(e)
                if "syntax error" in error_msg.lower():
                    error_msg = f"SQL语法错误: {error_msg}"
                elif "does not exist" in error_msg.lower():
                    error_msg = f"表或字段不存在: {error_msg}"
                return {"error": error_msg}
        
        # 所有尝试都失败
        logger.error("PDConnection.sql timeout for all attempts!")
        return {"error": "查询超时，请稍后重试"}

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.closeall()