import logging
import os
import re
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from typing import List, Dict, Tuple, Generator, Any, Callable, Optional, Union

from deepdoc.parser.utils import get_text
from rag.utils import num_tokens_from_string


class RAGFlowCodeParser:
    """Parser for code files with tree-sitter support"""
    
    def __init__(self):
        # 初始化Python解析器和语言
        self.PY_LANGUAGE = Language(tspython.language())
        # 在新版API中直接通过构造函数设置语言
        self.parser = Parser(self.PY_LANGUAGE)
        self.max_chunk_size = 512
        self.min_chunk_size = 100  # 最小块大小，小于此值的块将被合并
        
        # 定义节点类型到处理函数的映射
        self.node_type_handlers = {
            'function_definition': self._handle_function_definition,
            'class_definition': self._handle_class_definition,
            'import_statement': self._handle_import_statement,
            'import_from_statement': self._handle_import_statement,
            'module': self._handle_small_node,  # 处理整个模块
        }
        
        # 保存已处理的文件模块导入信息
        self.module_imports = {}
    
    def collapsed_replacement(self, node_type: str) -> str:
        """返回节点折叠后的替换文本
        
        Args:
            node_type: 节点类型
            
        Returns:
            替换文本
        """
        replacements = {
            'block': ":\n    ...",
            'class_body': " {...}",
            'statement_block': ":\n    ...",
            'default': "..."
        }
        return replacements.get(node_type, replacements['default'])
    
    def __call__(self, filename, binary=None, max_chunk_size=512, **kwargs):
        """Parse code file into chunks

        Args:
            filename: File name or path
            binary: Binary content of the file
            max_chunk_size: Maximum token size for each chunk
            **kwargs: Additional arguments

        Returns:
            List of code chunks
        """
        self.max_chunk_size = max_chunk_size
        code_text = get_text(filename, binary)
        file_path = filename if isinstance(filename, str) else kwargs.get("file_path", "unknown")
        
        # 检测文件类型
        file_extension = os.path.splitext(file_path)[1].lower() if isinstance(file_path, str) else '.py'
        
        # 目前仅支持Python
        if file_extension in ['.py', '.pyx', '.pyi', '.pyw']:
            return self._parse_python_code(code_text, file_path)
        else:
            # 对于不支持的文件类型，使用简单分割
            return self._fallback_parsing(code_text)
    
    def _parse_python_code(self, code_text: str, file_path: str) -> List[List[Any]]:
        """Parse Python code using tree-sitter

        Args:
            code_text: Python code text
            file_path: Path of the code file

        Returns:
            List of code chunks with their metadata
        """
        code_bytes = code_text.encode('utf-8')
        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node
        
        # 提取导入语句作为单独的块
        import_chunks = self._extract_imports(root_node, code_bytes, file_path)
        
        # 从生成器获取所有代码块
        chunks = list(self._get_smart_collapsed_chunks(root_node, code_bytes, file_path))
        
        # 合并小块
        chunks = self._merge_small_chunks(chunks)
        
        # 如果导入语句块存在，将其添加到结果前面
        if import_chunks:
            chunks = import_chunks + chunks
        
        logging.info(f"Code parser found {len(chunks)} chunks for {file_path}")
        
        # 如果没有找到有意义的代码块，则使用备用解析
        if not chunks:
            return self._fallback_parsing(code_text)
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[List[Any]], min_chunk_size: int = None) -> List[List[Any]]:
        """合并过小的代码块
        
        Args:
            chunks: 代码块列表
            min_chunk_size: 最小块大小(token数)，如果为None则使用self.min_chunk_size
            
        Returns:
            合并后的代码块列表
        """
        if not chunks:
            return []
        
        if min_chunk_size is None:
            min_chunk_size = self.min_chunk_size
        
        # 候选合并块
        result = []
        current_chunk = None
        current_tokens = 0
        current_path = ""
        
        # 预处理：识别类定义块和它们的方法块，优先保持它们的连续性
        class_blocks = {}  # 类名 -> (类块索引, 开始行)
        method_blocks = {}  # 类名 -> [方法块索引]
        
        for i, (chunk_text, path) in enumerate(chunks):
            # 检测类定义块
            if chunk_text.strip().startswith('class ') and ':' in chunk_text.split('\n')[0]:
                class_name = path.split('.')[-1] if '.' in path else path
                class_blocks[class_name] = (i, chunk_text.split('\n')[0])
            # 检测方法块
            elif '.' in path:
                class_name = path.split('.')[-2] if path.count('.') > 1 else path.split('.')[0]
                if class_name not in method_blocks:
                    method_blocks[class_name] = []
                method_blocks[class_name].append(i)
        
        # 处理所有块
        i = 0
        while i < len(chunks):
            chunk_text, path = chunks[i]
            chunk_tokens = num_tokens_from_string(chunk_text)
            
            # 特殊处理：如果是类定义块且有对应的方法块，尝试合并
            class_name = path.split('.')[-1] if '.' in path else path
            if class_name in class_blocks and class_blocks[class_name][0] == i and class_name in method_blocks:
                # 尝试合并类和它的方法
                merged_chunk = chunk_text
                merged_tokens = chunk_tokens
                merged_indices = [i]
                
                for method_idx in sorted(method_blocks[class_name]):
                    if method_idx > i:  # 只考虑当前块之后的方法块
                        method_text, method_path = chunks[method_idx]
                        method_tokens = num_tokens_from_string(method_text)
                        
                        # 检查合并后是否超过大小限制
                        separator = "\n\n" if path == method_path else "\n"
                        if merged_tokens + method_tokens + num_tokens_from_string(separator) <= self.max_chunk_size:
                            merged_chunk += separator + method_text
                            merged_tokens += method_tokens + num_tokens_from_string(separator)
                            merged_indices.append(method_idx)
                        else:
                            break
                
                # 如果成功合并了方法，添加合并后的块并跳过已处理的索引
                if len(merged_indices) > 1:
                    result.append([merged_chunk, path])
                    i = max(merged_indices) + 1
                    continue
            
            # 正常处理逻辑
            if chunk_tokens >= min_chunk_size:
                # 如果有累积的小块，先添加它
                if current_chunk:
                    result.append([current_chunk, current_path])
                    current_chunk = None
                    current_tokens = 0
                
                result.append([chunk_text, path])
            else:
                # 处理小块
                if not current_chunk:
                    # 开始新的累积块
                    current_chunk = chunk_text
                    current_tokens = chunk_tokens
                    current_path = path
                elif current_tokens + chunk_tokens <= self.max_chunk_size:
                    # 可以合并到当前累积块
                    # 添加分隔符，避免使用 # Path: 注释，它可能导致代码结构被破坏
                    if current_path == path:
                        separator = "\n\n"
                    else:
                        # 如果路径看起来是类.方法结构，使用更智能的分隔
                        if '.' in current_path and '.' in path and current_path.split('.')[0] == path.split('.')[0]:
                            separator = "\n\n"
                        else:
                            separator = "\n\n# Path: " + path + "\n"
                    
                    current_chunk += separator + chunk_text
                    current_tokens += chunk_tokens + num_tokens_from_string(separator)
                else:
                    # 当前累积块已满，添加到结果并开始新块
                    result.append([current_chunk, current_path])
                    current_chunk = chunk_text
                    current_tokens = chunk_tokens
                    current_path = path
            
            i += 1
        
        # 添加最后的累积块
        if current_chunk:
            result.append([current_chunk, current_path])
        
        return result
    
    def _extract_imports(self, node, code_bytes: bytes, file_path: str) -> List[List[Any]]:
        """Extract import statements as a separate chunk
        
        Args:
            node: Root node of the AST
            code_bytes: Code as bytes
            file_path: Path of the code file
            
        Returns:
            List containing the import statements chunk if found
        """
        import_statements = []
        
        # 查找所有导入语句
        for child in node.children:
            if child.type in ('import_statement', 'import_from_statement'):
                import_text = code_bytes[child.start_byte:child.end_byte].decode()
                import_statements.append(import_text)
        
        # 如果有导入语句，创建一个块
        if import_statements:
            import_block = "# Imports\n" + "\n".join(import_statements)
            # 存储模块导入信息以便后续使用
            self.module_imports[file_path] = import_block
            return [[import_block, f"{file_path}:imports"]]
        
        return []
    
    def _collapse_children(self, node, code_bytes: bytes, block_types: List[str], collapse_types: List[str], 
                         collapse_block_types: List[str], max_chunk_size: int) -> str:
        """折叠节点的子节点，保持整体结构但减小大小
        
        Args:
            node: 父节点
            code_bytes: 完整代码字节
            block_types: 表示块体的节点类型列表(如 'block', 'class_body')
            collapse_types: 需要折叠的子节点类型列表(如 'function_definition')
            collapse_block_types: 在折叠节点中需要替换的块类型(如 'block')
            max_chunk_size: 最大块大小
            
        Returns:
            折叠后的代码文本
        """
        # 提取节点的完整代码
        code = code_bytes[node.start_byte:node.end_byte].decode()
        
        # 查找表示块体的节点
        block_node = None
        for child in node.children:
            if child.type in block_types:
                block_node = child
                break
        
        if not block_node:
            return code
        
        # 收集需要折叠的子节点
        collapsed_children = []
        children_to_collapse = []
        
        for child in block_node.children:
            if child.type in collapse_types:
                children_to_collapse.append(child)
        
        # 为需要折叠的节点创建折叠版本
        for child in reversed(children_to_collapse):
            # 查找子节点中的块节点
            grand_child = None
            for gc in child.children:
                if gc.type in collapse_block_types:
                    grand_child = gc
                    break
            
            if grand_child:
                # 提取块之前的代码
                prefix = code_bytes[child.start_byte:grand_child.start_byte].decode()
                # 清理前缀中可能的问题字符
                if prefix.startswith('d '):
                    prefix = prefix[2:]
                # 创建折叠的子节点
                replacement = self.collapsed_replacement(grand_child.type)
                collapsed_child = prefix + replacement
                # 收集折叠后的代码
                collapsed_children.insert(0, (child, collapsed_child, grand_child))
        
        # 应用折叠：先创建完整的折叠版本
        folded_code = code
        for child, collapsed_text, grand_child in collapsed_children:
            # 计算原始块的范围
            start = grand_child.start_byte - node.start_byte
            end = grand_child.end_byte - node.start_byte
            
            if start >= 0 and end <= len(folded_code.encode()):
                # 替换原始块为折叠版本
                folded_bytes = folded_code.encode()
                prefix = folded_bytes[:start].decode()
                suffix = folded_bytes[end:].decode()
                replacement_text = self.collapsed_replacement(grand_child.type)
                folded_code = prefix + replacement_text + suffix
        
        # 检查是否仍然太大，如果是，使用更合适的折叠方式
        if num_tokens_from_string(folded_code) > max_chunk_size:
            # 检查节点类型，使用更专门的折叠方法
            if node.type == 'class_definition':
                return self._collapse_class(node, code_bytes)
            elif node.type == 'function_definition':
                return self._collapse_function(node, code_bytes)
            
            # 如果不是类或函数，只能移除一些子节点，但保留它们的签名
            removed_child = False
            while num_tokens_from_string(folded_code) > max_chunk_size and collapsed_children:
                removed_child = True
                # 从末尾开始移除子节点
                child, collapsed_text, grand_child = collapsed_children.pop()
                # 获取子节点类型
                child_type = child.type
                
                if child_type == 'function_definition':
                    # 对于函数，只保留签名，不完全移除
                    # 查找函数体
                    func_body = None
                    for func_child in child.children:
                        if func_child.type in collapse_block_types:
                            func_body = func_child
                            break
                    
                    if func_body:
                        # 获取函数签名
                        child_start = child.start_byte - node.start_byte
                        body_start = func_body.start_byte - node.start_byte
                        child_end = child.end_byte - node.start_byte
                        
                        if child_start >= 0 and child_end <= len(folded_code.encode()):
                            folded_bytes = folded_code.encode()
                            # 提取前缀(包括函数签名)和后缀
                            prefix = folded_bytes[:body_start].decode()
                            suffix = folded_bytes[child_end:].decode()
                            # 替换整个函数为"函数签名 + ..."
                            folded_code = prefix + ":\n    ..." + suffix
                else:
                    # 其他节点类型，保守地移除整个子节点
                    child_start = child.start_byte - node.start_byte
                    child_end = child.end_byte - node.start_byte
                    
                    if child_start >= 0 and child_end <= len(folded_code.encode()):
                        folded_bytes = folded_code.encode()
                        prefix = folded_bytes[:child_start].decode()
                        suffix = folded_bytes[child_end:].decode()
                        folded_code = prefix + suffix
            
            # 如果移除了子节点，清理多余的空行
            if removed_child:
                lines = folded_code.split('\n')
                cleaned_lines = []
                prev_empty = False
                
                for line in lines:
                    if line.strip() == '':
                        if not prev_empty:
                            cleaned_lines.append(line)
                        prev_empty = True
                    else:
                        cleaned_lines.append(line)
                        prev_empty = False
                
                folded_code = '\n'.join(cleaned_lines)
        
        return folded_code
    
    def _get_smart_collapsed_chunks(self, node, code_bytes: bytes, file_path: str, path: str = "") -> Generator[List[Any], None, None]:
        """Get code chunks with smart collapsing
        
        Args:
            node: Tree-sitter node
            code_bytes: Code as bytes
            file_path: Path of the code file
            path: Current code path (class.method etc.)
            
        Yields:
            Code chunks with their metadata
        """
        # 1. 如果是小节点且内容有价值，直接作为一个块
        if self._is_small_valuable_node(node, code_bytes):
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
            if chunk_text.strip():  # 确保不是空文本
                yield [chunk_text, path if path else file_path]
                # 注意: 即使产生了这个节点的块，也继续处理其子节点
        
        # 2. 如果节点类型有专门的处理器，调用它
        if node.type in self.node_type_handlers:
            handler = self.node_type_handlers[node.type]
            chunk = handler(node, code_bytes, file_path, path)
            if chunk:
                yield chunk
        
        # 3. 递归处理所有子节点
        for child in node.children:
            current_path = path
            
            # 更新路径信息
            if child.type in ('function_definition', 'class_definition'):
                name_node = self._get_name_node(child)
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode()
                    current_path = f"{path}.{name}" if path else name
            
            # 递归处理子节点
            yield from self._get_smart_collapsed_chunks(child, code_bytes, file_path, current_path)
    
    def _is_small_valuable_node(self, node, code_bytes: bytes) -> bool:
        """Check if node is small enough to be kept entirely
        
        Args:
            node: Tree-sitter node
            code_bytes: Code as bytes
            
        Returns:
            True if node is small and valuable
        """
        # 这些节点类型通常是有价值的小节点
        valuable_types = [
            'import_statement', 
            'import_from_statement',
            'expression_statement',
            'assignment',
            'global_statement',
            'comment',
            'docstring'
        ]
        
        if node.type in valuable_types:
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
            return num_tokens_from_string(chunk_text) <= self.max_chunk_size
        
        return False
    
    def _get_name_node(self, node) -> Optional[Any]:
        """Get the name node of a function or class
        
        Args:
            node: Tree-sitter node
            
        Returns:
            Name node if found, None otherwise
        """
        # 先尝试通过field_name获取
        for field_name in ['name', 'identifier']:
            try:
                name_node = node.child_by_field_name(field_name)
                if name_node:
                    return name_node
            except:
                pass
        
        # 如果失败，遍历所有子节点查找identifier
        for child in node.children:
            if child.type == 'identifier':
                return child
        
        return None
    
    def _handle_function_definition(self, node, code_bytes: bytes, file_path: str, path: str) -> List[Any]:
        """Handle function definition node
        
        Args:
            node: Function definition node
            code_bytes: Code as bytes
            file_path: Path of the code file
            path: Current code path
            
        Returns:
            Function chunk
        """
        chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
        
        # 如果整个函数小于最大块大小，直接返回完整函数
        if num_tokens_from_string(chunk_text) <= self.max_chunk_size:
            return [chunk_text, path if path else file_path]
        
        # 否则使用智能折叠算法
        collapsed = self._collapse_children(
            node,
            code_bytes,
            ['block'],  # 函数体节点类型
            [],  # 函数内部没有需要特别折叠的子节点
            ['block'],  # 要替换的块类型
            self.max_chunk_size
        )
        
        # 如果折叠后的代码仍然过大，使用更简单的折叠
        if num_tokens_from_string(collapsed) > self.max_chunk_size:
            collapsed = self._collapse_function(node, code_bytes)
        
        return [collapsed, path if path else file_path]
    
    def _handle_class_definition(self, node, code_bytes: bytes, file_path: str, path: str) -> List[Any]:
        """Handle class definition node
        
        Args:
            node: Class definition node
            code_bytes: Code as bytes
            file_path: Path of the code file
            path: Current code path
            
        Returns:
            Class chunk
        """
        chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
        
        # 如果整个类小于最大块大小，直接返回完整类
        if num_tokens_from_string(chunk_text) <= self.max_chunk_size:
            return [chunk_text, path if path else file_path]
        
        # 否则使用智能折叠算法
        collapsed = self._collapse_children(
            node,
            code_bytes,
            ['block'],  # 类体节点类型
            ['function_definition'],  # 要折叠的子节点类型(方法)
            ['block'],  # 要替换的块类型
            self.max_chunk_size
        )
        
        # 如果折叠后的代码仍然过大，使用更简单的折叠
        if num_tokens_from_string(collapsed) > self.max_chunk_size:
            collapsed = self._collapse_class(node, code_bytes)
        
        return [collapsed, path if path else file_path]
    
    def _handle_import_statement(self, node, code_bytes: bytes, file_path: str, path: str) -> Optional[List[Any]]:
        """Handle import statement node
        
        Args:
            node: Import statement node
            code_bytes: Code as bytes
            file_path: Path of the code file
            path: Current code path
            
        Returns:
            Import chunk or None if already handled by _extract_imports
        """
        # 通常导入语句已经由_extract_imports处理，这里返回None
        # 但如果发现了一个独立的、尚未分组的导入语句，可以单独处理
        if node.parent and node.parent.type != 'module':
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
            return [chunk_text, f"{file_path}:import"]
        return None
    
    def _handle_small_node(self, node, code_bytes: bytes, file_path: str, path: str) -> Optional[List[Any]]:
        """Handle small nodes that should be preserved entirely
        
        Args:
            node: Tree-sitter node
            code_bytes: Code as bytes
            file_path: Path of the code file
            path: Current code path
            
        Returns:
            Node chunk or None
        """
        chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
        if num_tokens_from_string(chunk_text) <= self.max_chunk_size:
            return [chunk_text, path if path else file_path]
        return None
    
    def _collapse_class(self, node, code_bytes: bytes) -> str:
        """Collapse class definition to fit within token limit
        
        Args:
            node: Class node
            code_bytes: Code as bytes
            
        Returns:
            Collapsed class definition
        """
        # 查找类体
        body_node = None
        for child in node.children:
            if child.type == 'block':
                body_node = child
                break
        
        if not body_node:
            return code_bytes[node.start_byte:node.end_byte].decode()
        
        # 获取类定义和类体开始
        class_def = code_bytes[node.start_byte:body_node.start_byte].decode().rstrip()
        
        # 确保类定义结尾没有多余字符
        if not class_def.endswith(':'):
            class_def = class_def.rstrip() + ':'
        
        # 收集方法签名
        method_signatures = []
        for child in body_node.children:
            if child.type == 'function_definition':
                # 查找函数体
                func_body = None
                for func_child in child.children:
                    if func_child.type == 'block':
                        func_body = func_child
                        break
                
                if func_body:
                    # 获取方法签名，确保没有多余字符
                    method_sig = code_bytes[child.start_byte:func_body.start_byte].decode().strip()
                    if method_sig.startswith('d '):  # 修复可能的错误前缀
                        method_sig = method_sig[2:]
                    method_signatures.append("    " + method_sig + ":\n        ...")
        
        # 构建折叠后的类定义
        collapsed_class = f"{class_def}\n"
        collapsed_class += "\n".join(method_signatures)
        
        # 如果没有方法，添加占位符
        if not method_signatures:
            collapsed_class += "\n    pass"
        
        return collapsed_class
    
    def _collapse_function(self, node, code_bytes: bytes) -> str:
        """Collapse function definition to fit within token limit
        
        Args:
            node: Function node
            code_bytes: Code as bytes
            
        Returns:
            Collapsed function definition
        """
        # 查找函数体
        body_node = None
        for child in node.children:
            if child.type == 'block':
                body_node = child
                break
        
        if not body_node:
            return code_bytes[node.start_byte:node.end_byte].decode()
        
        # 获取函数签名
        func_sig = code_bytes[node.start_byte:body_node.start_byte].decode().rstrip()
        
        # 确保函数签名结尾没有多余字符
        if not func_sig.endswith(':'):
            func_sig = func_sig.rstrip() + ':'
        
        # 构建折叠后的函数定义
        collapsed_func = f"{func_sig}\n    ..."
        
        return collapsed_func
    
    def _fallback_parsing(self, code_text: str) -> List[List[Any]]:
        """Fallback parsing for unsupported file types or when tree-sitter fails
        
        Args:
            code_text: Code text
            
        Returns:
            List of code chunks
        """
        chunks = []
        lines = code_text.split('\n')
        
        # 使用简单的基于代码行的分块策略
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = num_tokens_from_string(line)
            
            # 如果单独一行超过最大token数，则单独作为一个块
            if line_tokens > self.max_chunk_size:
                if current_chunk:
                    chunks.append(['\n'.join(current_chunk), ""])
                    current_chunk = []
                    current_tokens = 0
                
                # 分割过长的行
                words = line.split()
                temp_line = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = num_tokens_from_string(word)
                    if temp_tokens + word_tokens <= self.max_chunk_size:
                        temp_line.append(word)
                        temp_tokens += word_tokens
                    else:
                        chunks.append([' '.join(temp_line), ""])
                        temp_line = [word]
                        temp_tokens = word_tokens
                
                if temp_line:
                    chunks.append([' '.join(temp_line), ""])
            
            # 常规情况下添加行到当前块
            elif current_tokens + line_tokens <= self.max_chunk_size:
                current_chunk.append(line)
                current_tokens += line_tokens
            else:
                chunks.append(['\n'.join(current_chunk), ""])
                current_chunk = [line]
                current_tokens = line_tokens
        
        # 添加最后的块
        if current_chunk:
            chunks.append(['\n'.join(current_chunk), ""])
        
        # 清理空行
        chunks = self._clean_empty_lines(chunks)
        
        # 合并小块
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _clean_empty_lines(self, chunks: List[List[Any]]) -> List[List[Any]]:
        """Clean excessive empty lines in chunk text
        
        Args:
            chunks: List of chunks
            
        Returns:
            Cleaned chunks
        """
        cleaned_chunks = []
        
        for chunk_text, path in chunks:
            lines = chunk_text.split('\n')
            
            # 移除多余的空行
            cleaned_lines = []
            prev_empty = False
            
            for line in lines:
                if line.strip() == '':
                    if not prev_empty:
                        cleaned_lines.append(line)
                    prev_empty = True
                else:
                    cleaned_lines.append(line)
                    prev_empty = False
            
            cleaned_chunks.append(['\n'.join(cleaned_lines), path])
        
        return cleaned_chunks


# 别名以与项目中其他解析器保持一致
CodeParser = RAGFlowCodeParser
