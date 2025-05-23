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
import copy

ATTEMPT_TIME = 2
logger = logging.getLogger('ragflow.pd_conn')


@singleton
class PDConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        logger.info(f"Use ParadeDB {settings.PARADEDB['host']} as the doc engine.")

        # 连接池状态监控变量
        self._total_acquired = 0
        self._total_released = 0
        self._last_connection_check = time.time()
        self._connection_check_interval = 60  # 每60秒检查一次连接状态

        for _ in range(ATTEMPT_TIME):
            try:
                # 创建连接池 - 增加最大连接数
                self.pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=settings.PARADEDB.get("max_connections", 50),  # 默认提高到50
                    host=settings.PARADEDB["host"],
                    port=settings.PARADEDB["port"],
                    database=settings.PARADEDB.get("database", settings.PARADEDB.get("name", "rag_flow")),
                    user=settings.PARADEDB["user"],
                    password=settings.PARADEDB["password"]
                )

                # 测试连接
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT version()")
                        self.info["version"] = cur.fetchone()[0]
                finally:
                    # 确保测试连接归还到池中
                    self._return_connection(conn)
                break
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting ParadeDB {settings.PARADEDB['host']} to be healthy.")
                time.sleep(5)

        if not self.info:
            msg = f"ParadeDB {settings.PARADEDB['host']} is unhealthy in 120s."
            logger.error(msg)
            raise Exception(msg)

        logger.info(f"ParadeDB {settings.PARADEDB['host']} is healthy.")

    def _get_connection(self):
        """获取连接并记录连接计数"""
        self._check_connection_pool()
        try:
            conn = self.pool.getconn()
            self._total_acquired += 1
            return ManagedConnection(self, conn)
        except psycopg2.pool.PoolError as e:
            # 如果池已耗尽，尝试重置连接池
            logger.error(f"Connection pool error: {str(e)}. Trying to reset connections.")
            self._reset_connection_pool()
            return ManagedConnection(self, self.pool.getconn())

    def _return_connection(self, conn):
        """归还连接并记录连接计数"""
        if conn:
            try:
                # 检查连接是否在已使用的连接池中
                if conn in self.pool._used:
                    self.pool.putconn(conn)
                    self._total_released += 1
                else:
                    logger.warning("Attempting to return unkeyed connection to pool")
            except Exception as e:
                logger.error(f"Error returning connection to pool: {str(e)}")

    def _check_connection_pool(self):
        """检查连接池状态，如果发现泄漏则尝试修复"""
        now = time.time()
        if now - self._last_connection_check > self._connection_check_interval:
            self._last_connection_check = now
            
            # 计算未释放的连接数
            leaked = self._total_acquired - self._total_released
            
            # 记录连接池状态
            logger.info(f"Connection pool status: acquired={self._total_acquired}, released={self._total_released}, leaked={leaked}, used={self.pool._used}")
            
            # 如果泄漏连接太多，尝试重置连接池
            if leaked > 10:
                logger.warning(f"Detected {leaked} leaked connections. Attempting to reset connection pool.")
                self._reset_connection_pool()

    def _reset_connection_pool(self):
        """尝试关闭所有连接并重置连接池"""
        try:
            # 关闭所有连接
            self.pool.closeall()
            
            # 创建新的连接池
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=settings.PARADEDB.get("max_connections", 50),
                host=settings.PARADEDB["host"],
                port=settings.PARADEDB["port"],
                database=settings.PARADEDB.get("database", settings.PARADEDB.get("name", "rag_flow")),
                user=settings.PARADEDB["user"],
                password=settings.PARADEDB["password"]
            )
            
            # 重置计数器
            self._total_acquired = 0
            self._total_released = 0
            
            logger.info("Connection pool has been reset successfully.")
        except Exception as e:
            logger.error(f"Failed to reset connection pool: {str(e)}")

    """
    Database operations
    """

    def dbType(self) -> str:
        return "paradedb"

    def health(self) -> dict:
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                health_dict = dict(cur.fetchone())
                health_dict["type"] = "paradedb"
                # 添加连接池状态信息
                health_dict["pool_acquired"] = self._total_acquired
                health_dict["pool_released"] = self._total_released
                health_dict["pool_used"] = len(self.pool._used)
                return health_dict
        except Exception as e:
            logger.exception(f"PDConnection.health got exception: {str(e)}")
            return {"type": "paradedb", "status": "error", "error": str(e)}
        finally:
            if conn:
                self._return_connection(conn)

    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        # 检查表是否已存在
        if self.indexExist(indexName):
            logger.info(f"Table {indexName} already exists, checking schema...")
            # 升级表结构以确保所有字段都存在
            if self._upgrade_table_schema(indexName):
                logger.info(f"Table {indexName} schema verified/upgraded successfully")
                return True
            else:
                logger.error(f"Failed to verify/upgrade table {indexName} schema")
                return False
        
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # 创建更完善的表结构，包含与ES兼容的字段
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {indexName} (
                        id TEXT PRIMARY KEY,
                        kb_id TEXT NOT NULL,
                        doc_id TEXT,                 -- 文档ID
                        docnm_kwd TEXT,              -- 文档名称(与ES兼容)
                        content TEXT,                -- 原始内容
                        content_ltks TEXT,           -- 内容标记(与ES兼容)
                        content_sm_ltks TEXT,        -- 内容细粒度标记(与ES兼容)
                        content_with_weight TEXT,    -- 带权重内容(与ES兼容)
                        title TEXT,                  -- 标题
                        title_tks TEXT,              -- 标题标记(与ES兼容)
                        title_sm_tks TEXT,           -- 标题细粒度标记(与ES兼容)
                        name_kwd TEXT,               -- 名称关键词(与ES兼容)
                        important_kwd JSONB,         -- 重要关键词(与ES兼容)
                        important_tks TEXT,          -- 重要关键词标记(与ES兼容)
                        question_kwd JSONB,          -- 问题关键词(与ES兼容)
                        question_tks TEXT,           -- 问题关键词标记(与ES兼容)
                        img_id TEXT,                 -- 图片ID(与ES兼容)
                        position_int JSONB,          -- 位置信息(与ES兼容，支持数组)
                        page_num_int JSONB,          -- 页码(与ES兼容，支持数组)
                        top_int JSONB,               -- 顶部位置(与ES兼容，支持数组)
                        available_int INTEGER DEFAULT 1,  -- 可用标志
                        create_timestamp_flt FLOAT,  -- 创建时间戳(与ES兼容)
                        create_time TEXT,            -- 创建时间字符串
                        authors_tks TEXT,            -- 作者标记(与ES兼容)
                        authors_sm_tks TEXT,         -- 作者细粒度标记(与ES兼容)
                        weight_int JSONB DEFAULT '0',     -- 权重整数(支持数组)
                        weight_flt JSONB DEFAULT '0.0',   -- 权重浮点(支持数组)
                        rank_int JSONB DEFAULT '0',       -- 排名整数(支持数组)
                        rank_flt JSONB DEFAULT '0.0',     -- 排名浮点(支持数组)
                        knowledge_graph_kwd TEXT,    -- 知识图谱关键词
                        entities_kwd TEXT,           -- 实体关键词
                        pagerank_fea INTEGER DEFAULT 0,   -- PageRank特征
                        tag_feas TEXT,               -- 标签特征
                        tag_kwd TEXT,                -- 标签关键词
                        from_entity_kwd TEXT,        -- 来源实体关键词
                        to_entity_kwd TEXT,          -- 目标实体关键词
                        entity_kwd TEXT,             -- 实体关键词
                        entity_type_kwd TEXT,        -- 实体类型关键词
                        source_id TEXT,              -- 源ID
                        n_hop_with_weight TEXT,      -- N跳带权重
                        removed_kwd TEXT,            -- 移除的关键词
                        metadata JSONB,              -- 其他元数据
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
                    CREATE INDEX IF NOT EXISTS {indexName}_bm25 ON {indexName}
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
                logger.info(f"Creating index with name: {indexName}")
                # 创建后验证
                if self.indexExist(indexName):
                    logger.info(f"Successfully created and verified index: {indexName}")
                else:
                    logger.error(f"Failed to verify index after creation: {indexName}")
                return True
        except Exception as e:
            logger.exception(f"PDConnection.createIdx error {indexName}: {str(e)}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # 删除BM25索引
                cur.execute(f"DROP INDEX IF EXISTS {indexName}_bm25")
                # 删除向量索引
                cur.execute(f"DROP INDEX IF EXISTS {indexName}_vector_idx")
                # 删除表
                cur.execute(f"DROP TABLE IF EXISTS {indexName}")
                conn.commit()
        except Exception as e:
            logger.exception(f"PDConnection.deleteIdx error {indexName}: {str(e)}")
        finally:
            if conn:
                self._return_connection(conn)

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (indexName,))
                return cur.fetchone()[0]
        except Exception as e:
            logger.exception(f"PDConnection.indexExist got exception: {str(e)}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

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
                # ParadeDB中使用固定的embedding字段名，无论原始字段名是什么
                # 原始字段名如q_1024_vec会在insert时映射到embedding字段

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
        # 确保SQL语法正确
        select_part = f"SELECT {', '.join(select_clause)}"
        from_part = f"FROM {indexNames[0]}"
        
        # 添加限制条件
        limit_clause = ""
        if has_vector_search:
            limit_clause = f"LIMIT {vector_topn}"
        elif limit > 0:
            limit_clause = f"LIMIT {limit}"
            
        offset_clause = f"OFFSET {offset}" if offset > 0 else ""
        order_by_clause = f"ORDER BY {', '.join(order_clause)}" if order_clause else ""
        
        # 构建完整SQL - 确保每个部分都有正确的关键字
        sql = f"""
            {select_part}
            {from_part}
            {where_clause}
            {order_by_clause}
            {limit_clause}
            {offset_clause}
        """
        
        logger.debug(f"Generated SQL: {sql}")

        conn = None
        try:
            conn = self._get_connection()
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
        except psycopg2.errors.UndefinedColumn as e:
            # 字段不存在的友好错误处理
            missing_field = str(e).split('"')[1] if '"' in str(e) else "unknown"
            logger.error(f"Missing column in ParadeDB: {missing_field}")
            return {
                "hits": {
                    "total": 0,
                    "hits": []
                },
                "error": f"字段 '{missing_field}' 不存在，请检查数据库结构或更新应用"
            }
        except Exception as e:
            logger.exception(f"PDConnection.search error: {str(e)}")
            raise e
        finally:
            # 确保连接被归还到连接池
            if conn:
                self._return_connection(conn)

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT * FROM {indexName} 
                    WHERE id = %s AND kb_id = ANY(%s)
                """, (chunkId, knowledgebaseIds))
                result = cur.fetchone()
                if result:
                    result_dict = dict(result)
                    
                    # 将embedding字段映射回动态的向量字段名以保持兼容性
                    if 'embedding' in result_dict and result_dict['embedding'] is not None:
                        vector_data = result_dict.pop('embedding')
                        # 根据向量长度生成动态字段名
                        vector_size = len(vector_data) if isinstance(vector_data, list) else len(vector_data.tolist())
                        result_dict[f'q_{vector_size}_vec'] = vector_data
                    
                    return result_dict
                return None
        except Exception as e:
            logger.exception(f"PDConnection.get({chunkId}) got exception: {str(e)}")
            return None
        finally:
            if conn:
                self._return_connection(conn)

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        errors = []
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                for doc in documents:
                    try:
                        # 确保文档有id字段
                        if "id" not in doc and "doc_id" in doc:
                            doc["id"] = doc["doc_id"]
                            
                        # 创建文档副本而不是修改原始文档
                        doc_copy = copy.deepcopy(doc)
                        
                        # 从副本中提取id
                        doc_id = doc_copy.pop("id", "")
                        doc_copy["kb_id"] = knowledgebaseId
                        
                        # 处理动态向量字段名映射到固定的embedding字段
                        vector_field_pattern = re.compile(r'q_(\d+)_vec')
                        embedding_data = None
                        for field_name in list(doc_copy.keys()):
                            match = vector_field_pattern.match(field_name)
                            if match:
                                # 找到向量字段，将其映射到embedding字段
                                embedding_data = doc_copy.pop(field_name)
                                break
                        
                        # 如果找到了向量数据，添加到embedding字段
                        if embedding_data is not None:
                            doc_copy["embedding"] = embedding_data
                        
                        # 处理可能包含数组或复杂数据的字段，确保它们能正确存储到JSONB字段中
                        jsonb_fields = {'position_int', 'page_num_int', 'top_int', 'weight_int', 'weight_flt', 'rank_int', 'rank_flt'}
                        for field_name in jsonb_fields:
                            if field_name in doc_copy:
                                value = doc_copy[field_name]
                                # 如果值不是None且不是字符串，将其转换为JSON字符串
                                if value is not None and not isinstance(value, str):
                                    try:
                                        import json
                                        doc_copy[field_name] = json.dumps(value)
                                    except (TypeError, ValueError) as e:
                                        # 如果无法序列化，转换为字符串
                                        doc_copy[field_name] = str(value)
                                        logger.warning(f"Field {field_name} value {value} converted to string: {e}")
                        
                        # 构建SQL的字段列表和占位符
                        fields = ", ".join(doc_copy.keys())
                        placeholders = ", ".join(["%s"] * len(doc_copy))
                        
                        # 构建UPDATE部分
                        update_clause = ", ".join([f"{k} = EXCLUDED.{k}" for k in doc_copy.keys()])

                        sql = f"""
                            INSERT INTO {indexName} (id, {fields})
                            VALUES (%s, {placeholders})
                            ON CONFLICT (id) DO UPDATE
                            SET {update_clause}
                        """

                        cur.execute(sql, [doc_id] + list(doc_copy.values()))
                    except Exception as e:
                        errors.append(f"{doc.get('id', 'unknown')}:{str(e)}")
                        logger.error(f"Insert error for doc {doc.get('id', 'unknown')}: {str(e)}")
                        conn.rollback()
                        continue

                conn.commit()
            return errors
        except Exception as e:
            logger.exception(f"PDConnection.insert got exception: {str(e)}")
            return [str(e)]
        finally:
            if conn:
                self._return_connection(conn)

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        conn = None
        try:
            conn = self._get_connection()
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
        except Exception as e:
            logger.exception(f"PDConnection.update got exception: {str(e)}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        conn = None
        try:
            conn = self._get_connection()
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
        except Exception as e:
            logger.exception(f"PDConnection.delete got exception: {str(e)}")
            return 0
        finally:
            if conn:
                self._return_connection(conn)

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
            
            # 处理向量字段的映射
            vector_field_requested = None
            vector_field_pattern = re.compile(r'q_(\d+)_vec')
            for field in fields:
                if vector_field_pattern.match(field):
                    vector_field_requested = field
                    break
            
            for field in fields:
                if field == vector_field_requested and 'embedding' in hit:
                    # 将embedding字段映射回请求的动态向量字段名
                    field_values[field] = hit['embedding']
                elif field in hit:
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
        
        # 处理向量字段名映射
        # 将q_*_vec字段映射到embedding字段
        vector_field_pattern = re.compile(r'\bq_\d+_vec\b')
        sql = vector_field_pattern.sub('embedding', sql)
        
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
            conn = None
            try:
                conn = self._get_connection()
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
            finally:
                if conn:
                    self._return_connection(conn)
        
        # 所有尝试都失败
        logger.error("PDConnection.sql timeout for all attempts!")
        return {"error": "查询超时，请稍后重试"}

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.closeall()

    def _upgrade_table_schema(self, indexName: str) -> bool:
        """
        升级现有表结构，添加缺失的字段
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # 获取当前表的字段列表
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (indexName,))
                existing_columns = {row[0] for row in cur.fetchall()}
                
                # 定义所有应该存在的字段及其类型
                required_fields = {
                    'content_sm_ltks': 'TEXT',
                    'title_sm_tks': 'TEXT',
                    'name_kwd': 'TEXT',
                    'important_tks': 'TEXT',
                    'question_tks': 'TEXT',
                    'create_time': 'TEXT',
                    'authors_tks': 'TEXT',
                    'authors_sm_tks': 'TEXT',
                    'weight_int': 'JSONB DEFAULT \'0\'',
                    'weight_flt': 'JSONB DEFAULT \'0.0\'',
                    'rank_int': 'JSONB DEFAULT \'0\'',
                    'rank_flt': 'JSONB DEFAULT \'0.0\'',
                    'knowledge_graph_kwd': 'TEXT',
                    'entities_kwd': 'TEXT',
                    'pagerank_fea': 'INTEGER DEFAULT 0',
                    'tag_feas': 'TEXT',
                    'tag_kwd': 'TEXT',
                    'from_entity_kwd': 'TEXT',
                    'to_entity_kwd': 'TEXT',
                    'entity_kwd': 'TEXT',
                    'entity_type_kwd': 'TEXT',
                    'source_id': 'TEXT',
                    'n_hop_with_weight': 'TEXT',
                    'removed_kwd': 'TEXT'
                }
                
                # 添加缺失的字段
                added_fields = []
                modified_fields = []
                
                # 需要修改类型的字段映射（从旧类型到新类型）
                type_changes = {
                    'page_num_int': ('integer', 'JSONB'),
                    'top_int': ('integer', 'JSONB'), 
                    'position_int': ('integer', 'JSONB'),
                    'weight_int': ('integer', 'JSONB'),
                    'weight_flt': ('double precision', 'JSONB'),
                    'rank_int': ('integer', 'JSONB'),
                    'rank_flt': ('double precision', 'JSONB')
                }
                
                # 检查需要修改类型的字段
                for field_name, (old_type, new_type) in type_changes.items():
                    if field_name in existing_columns:
                        # 检查当前字段类型
                        cur.execute(f"""
                            SELECT data_type 
                            FROM information_schema.columns 
                            WHERE table_name = %s AND column_name = %s
                        """, (indexName, field_name))
                        current_type_result = cur.fetchone()
                        if current_type_result and current_type_result[0] == old_type:
                            try:
                                # 修改字段类型
                                cur.execute(f"""
                                    ALTER TABLE {indexName} 
                                    ALTER COLUMN {field_name} TYPE {new_type} 
                                    USING {field_name}::text::{new_type}
                                """)
                                modified_fields.append(f"{field_name}({old_type}->{new_type})")
                                logger.info(f"Modified field {field_name} type from {old_type} to {new_type} in table {indexName}")
                            except Exception as e:
                                logger.warning(f"Failed to modify field {field_name} type in table {indexName}: {str(e)}")
                
                for field_name, field_type in required_fields.items():
                    if field_name not in existing_columns:
                        try:
                            cur.execute(f"ALTER TABLE {indexName} ADD COLUMN {field_name} {field_type}")
                            added_fields.append(field_name)
                            logger.info(f"Added missing field {field_name} to table {indexName}")
                        except Exception as e:
                            logger.warning(f"Failed to add field {field_name} to table {indexName}: {str(e)}")
                
                if added_fields or modified_fields:
                    conn.commit()
                    if added_fields:
                        logger.info(f"Successfully upgraded table {indexName}, added fields: {added_fields}")
                    if modified_fields:
                        logger.info(f"Successfully upgraded table {indexName}, modified fields: {modified_fields}")
                    return True
                else:
                    logger.info(f"Table {indexName} schema is up to date")
                    return True
                    
        except Exception as e:
            logger.exception(f"Failed to upgrade table {indexName} schema: {str(e)}")
            return False
        finally:
            if conn:
                self._return_connection(conn)

class ManagedConnection:
    """包装PostgreSQL连接，实现上下文管理器接口"""
    
    def __init__(self, connection_manager, conn):
        self.connection_manager = connection_manager
        self.conn = conn
        
    def __enter__(self):
        return self.conn
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection_manager._return_connection(self.conn)
        
    def cursor(self, *args, **kwargs):
        return self.conn.cursor(*args, **kwargs)
        
    def commit(self):
        return self.conn.commit()
        
    def rollback(self):
        return self.conn.rollback()