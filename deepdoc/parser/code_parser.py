import logging
import os
import re
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from typing import List, Dict, Tuple, Generator, Any, Callable, Optional, Union

from deepdoc.parser.utils import get_text
from rag.utils import num_tokens_from_string


class RAGFlowCodeParser:
    """Parser for code files with tree-sitter support for multiple languages"""
    
    def __init__(self, language: str = 'python'):
        """Initialize a code parser with support for multiple languages
        
        Args:
            language: Programming language to parse (default: 'python')
        """
        self.language = language
        self.max_chunk_size = 512
        self.min_chunk_size = 100  # 最小块大小，小于此值的块将被合并
        
        # 初始化解析器
        self.parser = self._initialize_parser(language)
        
        # 定义节点类型到处理函数的映射
        self.node_type_handlers = {
            'function_definition': self._handle_function_definition,
            'method_definition': self._handle_function_definition,  # 其他语言的方法定义
            'class_definition': self._handle_class_definition,
            'class': self._handle_class_definition,  # 其他语言的类定义
            'import_statement': self._handle_import_statement,
            'import_from_statement': self._handle_import_statement,
            'import_declaration': self._handle_import_statement,  # 其他语言的导入声明
            'module': self._handle_small_node,  # 处理整个模块
        }
        
        # 保存已处理的文件模块导入信息
        self.module_imports = {}
    
    def _initialize_parser(self, language: str) -> Parser:
        """初始化针对特定语言的解析器
        
        Args:
            language: 编程语言名称
            
        Returns:
            配置好的tree-sitter解析器
        """
        # 尝试使用tree_sitter_language_pack获取所有语言支持(包括Python)
        try:
            import tree_sitter_language_pack
            parser = tree_sitter_language_pack.get_parser(language)
            return parser
        except ImportError:
            logging.warning(
                "请安装tree_sitter_language_pack来获取多语言支持: "
                "pip install tree-sitter-language-pack"
            )
            # 尝试回退到Python内置支持
            if language.lower() == 'python':
                try:
                    py_language = Language(tspython.language())
                    parser = Parser(py_language)
                    return parser
                except Exception as e:
                    logging.error(f"无法初始化Python解析器: {e}")
                    raise ImportError("无法初始化解析器，请安装tree_sitter_language_pack")
            else:
                raise ImportError(
                    f"无法为语言 {language} 初始化解析器。请安装tree_sitter_language_pack。"
                    "查看 https://github.com/Goldziher/tree-sitter-language-pack#available-languages "
                    "获取支持的语言列表。"
                )
        except Exception as e:
            logging.warning(
                f"无法获取语言 {language} 的解析器。请检查 "
                "https://github.com/Goldziher/tree-sitter-language-pack#available-languages "
                "获取有效语言列表。"
            )
            # 尝试回退到Python内置支持
            if language.lower() == 'python':
                try:
                    py_language = Language(tspython.language())
                    parser = Parser(py_language)
                    return parser
                except Exception as ex:
                    logging.error(f"无法初始化Python解析器: {ex}")
                    raise ImportError("无法初始化解析器，请安装tree_sitter_language_pack")
            else:
                raise e
    
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
            'function_body': " {...}",  # 其他语言
            'method_body': " {...}",    # 其他语言
            'default': "..."
        }
        return replacements.get(node_type, replacements['default'])
    
    def __call__(self, filename, binary=None, max_chunk_size=512, language=None, **kwargs):
        """Parse code file into chunks

        Args:
            filename: File name or path
            binary: Binary content of the file
            max_chunk_size: Maximum token size for each chunk
            language: Override the language for this file
            **kwargs: Additional arguments

        Returns:
            List of code chunks
        """
        self.max_chunk_size = max_chunk_size
        
        # 如果指定了特定语言，使用该语言的解析器
        if language and language != self.language:
            original_parser = self.parser
            self.parser = self._initialize_parser(language)
            process_with_temp_parser = True
        else:
            process_with_temp_parser = False
        
        try:
            code_text = get_text(filename, binary)
            file_path = filename if isinstance(filename, str) else kwargs.get("file_path", "unknown")
            
            # 根据文件扩展名自动检测语言
            if isinstance(file_path, str):
                file_extension = os.path.splitext(file_path)[1].lower()
                detected_language = self._detect_language_from_extension(file_extension)
                
                # 如果检测到的语言与当前语言不同，切换解析器
                if detected_language and detected_language != self.language and not language:
                    logging.info(f"Detected language {detected_language} for {file_path}")
                    original_parser = self.parser
                    self.parser = self._initialize_parser(detected_language)
                    process_with_temp_parser = True
            
            # 解析代码文件
            chunks = self._parse_code_file(code_text, file_path)
            
            # 还原原始解析器
            if process_with_temp_parser:
                self.parser = original_parser
                
            return chunks
            
        except Exception as e:
            logging.error(f"Error parsing file {filename}: {e}")
            if process_with_temp_parser:
                self.parser = original_parser
            return self._fallback_parsing(code_text if 'code_text' in locals() else "")
    
    def _detect_language_from_extension(self, extension: str) -> Optional[str]:
        """根据文件扩展名检测编程语言
        
        Args:
            extension: 文件扩展名
            
        Returns:
            检测到的语言或None
        """
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'c_sharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.scala': 'scala',
            '.m': 'objective-c',
            '.lua': 'lua',
            '.r': 'r',
            '.sh': 'bash',
            '.pl': 'perl',
            '.jl': 'julia',
            '.dart': 'dart',
            '.elm': 'elm',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.erl': 'erlang',
            '.fs': 'f_sharp',
            '.fsx': 'f_sharp',
            '.hs': 'haskell',
            '.lhs': 'haskell',
            '.ml': 'ocaml',
            '.mli': 'ocaml',
        }
        return extension_map.get(extension)
    
    def _parse_code_file(self, code_text: str, file_path: str) -> List[List[Any]]:
        """Parse code file using appropriate parser
        
        Args:
            code_text: Code text
            file_path: Path of the code file
            
        Returns:
            List of code chunks with their metadata
        """
        try:
            code_bytes = code_text.encode('utf-8')
            tree = self.parser.parse(code_bytes)
            root_node = tree.root_node
            
            # 提取导入语句作为单独的块
            import_chunks = self._extract_imports(root_node, code_bytes, file_path)
            
            # 从生成器获取所有代码块
            chunks = list(self._recursive_chunk_node(root_node, code_bytes, file_path))
            
            # 去重和合并小块
            chunks = self._dedup_chunks(chunks)
            chunks = self._merge_small_chunks(chunks)
            
            # 如果导入语句块存在，将其添加到结果前面
            if import_chunks:
                chunks = import_chunks + chunks
            
            logging.info(f"Code parser found {len(chunks)} chunks for {file_path}")
            
            # 如果没有找到有意义的代码块，则使用备用解析
            if not chunks:
                return self._fallback_parsing(code_text)
            
            return chunks
        except Exception as e:
            logging.warning(f"Error parsing code file with tree-sitter: {e}")
            return self._fallback_parsing(code_text)
    
    # 递归分块方法
    def _recursive_chunk_node(self, node, code_bytes: bytes, file_path: str, path: str = "") -> Generator[List[Any], None, None]:
        """递归地将代码节点分块，保持语法结构完整性
        
        Args:
            node: Tree-sitter node
            code_bytes: 完整代码字节
            file_path: 文件路径
            path: 当前代码路径(类.方法)
            
        Yields:
            代码块及其元数据
        """
        # 适配不同语言的函数/类节点类型
        function_types = ('function_definition', 'method_definition', 'function_declaration', 'method_declaration')
        class_types = ('class_definition', 'class', 'class_declaration', 'struct_declaration')
        
        # 特殊处理类和函数定义
        if node.type in class_types or node.type in function_types:
            name_node = self._get_name_node(node)
            if name_node:
                name = code_bytes[name_node.start_byte:name_node.end_byte].decode()
                current_path = f"{path}.{name}" if path else name
                
                # 获取整个节点的代码
                node_code = code_bytes[node.start_byte:node.end_byte].decode()
                node_tokens = num_tokens_from_string(node_code)
                
                # 如果节点小于最大块大小，直接作为一个块
                if node_tokens <= self.max_chunk_size:
                    yield [node_code, current_path]
                    return  # 不再处理子节点
                elif node.type in class_types:
                    # 类太大，使用智能折叠
                    collapsed = self._collapse_class(node, code_bytes)
                    if num_tokens_from_string(collapsed) <= self.max_chunk_size:
                        yield [collapsed, current_path]
                        return
                elif node.type in function_types:
                    # 函数太大，使用智能折叠
                    collapsed = self._collapse_function(node, code_bytes)
                    if num_tokens_from_string(collapsed) <= self.max_chunk_size:
                        yield [collapsed, current_path]
                        return
        elif self._is_small_valuable_node(node, code_bytes):
            # 处理小的有价值节点
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
            if chunk_text.strip():
                yield [chunk_text, path if path else file_path]
                return
        
        # 创建当前块
        current_chunk = ""
        current_start = node.start_byte
        
        # 递归处理子节点
        for child in node.children:
            child_text = code_bytes[child.start_byte:child.end_byte].decode()
            child_tokens = num_tokens_from_string(child_text)
            
            # 当前子节点太大，递归处理
            if child_tokens > self.max_chunk_size:
                # 先添加已累积的内容（如果有）
                if current_chunk:
                    current_chunk_code = code_bytes[current_start:child.start_byte].decode()
                    if current_chunk_code.strip():
                        yield [current_chunk_code, path if path else file_path]
                    current_chunk = ""
                    current_start = child.end_byte
                
                # 根据子节点类型决定处理方式
                child_path = path
                if child.type in function_types or child.type in class_types:
                    name_node = self._get_name_node(child)
                    if name_node:
                        name = code_bytes[name_node.start_byte:name_node.end_byte].decode()
                        child_path = f"{path}.{name}" if path else name
                
                # 递归处理子节点
                yield from self._recursive_chunk_node(child, code_bytes, file_path, child_path)
            
            # 当前子节点会使累积块过大，添加现有块并开始新块
            elif current_chunk and num_tokens_from_string(current_chunk) + child_tokens > self.max_chunk_size:
                current_chunk_code = code_bytes[current_start:child.start_byte].decode()
                if current_chunk_code.strip():
                    yield [current_chunk_code, path if path else file_path]
                current_chunk = child_text
                current_start = child.start_byte
            # 累加到当前块
            else:
                current_chunk += child_text
        
        # 处理最后剩余的块
        if current_chunk:
            current_chunk_code = code_bytes[current_start:node.end_byte].decode()
            if current_chunk_code.strip():
                yield [current_chunk_code, path if path else file_path]
    
    def _dedup_chunks(self, chunks: List[List[Any]]) -> List[List[Any]]:
        """去除重复和高度重叠的代码块"""
        # 实现与之前相同
        # ...

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """计算两段文本的重叠比例"""
        # 防御性编程，避免空文本
        if not text1 or not text2:
            return 0.0
        
        # 处理文本以便比较
        lines1 = [line.strip() for line in text1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in text2.split('\n') if line.strip()]
        
        # 防止除零错误
        if not lines1 or not lines2:
            return 0.0
        
        # 计算共同行数
        common_lines = 0
        for line in lines1:
            if any(line == l2 for l2 in lines2):
                common_lines += 1
        
        # 返回重叠率
        overlap_ratio = common_lines / min(len(lines1), len(lines2))
        return overlap_ratio

    def _get_name_node(self, node) -> Optional[Any]:
        """获取函数或类的名称节点，兼容不同语言
        
        Args:
            node: Tree-sitter节点
            
        Returns:
            名称节点或None
        """
        # 常见的名称字段名列表，适用于不同语言
        field_names = ['name', 'identifier', 'id', 'Name', 'NAME']
        
        # 尝试通过字段名获取
        for field_name in field_names:
            try:
                name_node = node.child_by_field_name(field_name)
                if name_node:
                    return name_node
            except:
                pass
        
        # 尝试通过类型找到标识符节点
        identifier_types = ['identifier', 'Identifier', 'IDENTIFIER', 'simple_identifier', 'id']
        for child in node.children:
            if child.type in identifier_types:
                return child
            
            # 处理嵌套标识符的情况(如Java的完全限定名)
            for grandchild in child.children:
                if grandchild.type in identifier_types:
                    return grandchild
        
        return None
    
    def _is_small_valuable_node(self, node, code_bytes: bytes) -> bool:
        """检查节点是否足够小且有价值，兼容不同语言
        
        Args:
            node: Tree-sitter节点
            code_bytes: 代码字节
            
        Returns:
            True如果节点小且有价值
        """
        # 多语言支持的有价值节点类型
        valuable_types = [
            # Python
            'import_statement', 'import_from_statement',
            # JavaScript/TypeScript
            'import_declaration', 'export_declaration',
            # 多语言通用
            'expression_statement', 'assignment', 'declaration', 
            'variable_declaration', 'global_statement', 'comment',
            'docstring', 'string', 'documentation_comment'
        ]
        
        if node.type in valuable_types:
            chunk_text = code_bytes[node.start_byte:node.end_byte].decode()
            return num_tokens_from_string(chunk_text) <= self.max_chunk_size
        
        return False
    
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
            if child.type in ('import_statement', 'import_from_statement', 'import_declaration'):
                import_text = code_bytes[child.start_byte:child.end_byte].decode()
                import_statements.append(import_text)
        
        # 如果有导入语句，创建一个块
        if import_statements:
            import_block = "# Imports\n" + "\n".join(import_statements)
            # 存储模块导入信息以便后续使用
            self.module_imports[file_path] = import_block
            return [[import_block, f"{file_path}:imports"]]
        
        return []

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
        
        # 第一步：去除重复和重叠的代码块
        chunk_hashes = {}
        unique_chunks = []
        
        # 去重处理
        for i, (chunk_text, path) in enumerate(chunks):
            # 计算chunk的哈希值(基于内容)
            content_hash = hash(chunk_text.strip())
            
            if content_hash in chunk_hashes:
                # 如果是完全重复内容，与现有块比较
                existing_idx = chunk_hashes[content_hash]
                existing_text, existing_path = chunks[existing_idx]
                
                # 如果现有块较短或路径信息较少，用当前较完整的替换
                if len(chunk_text) > len(existing_text) or (len(path.split('.')) > len(existing_path.split('.'))):
                    unique_chunks[chunk_hashes[content_hash]] = [chunk_text, path]
                continue
            
            # 检查是否是其他块的子集或存在内容重叠
            is_subset = False
            for j, (other_text, other_path) in enumerate(unique_chunks):
                # 跳过空文本
                if not chunk_text.strip() or not other_text.strip():
                    continue
                
                # 检查内容重叠超过70%视为重复
                try:
                    overlap_ratio = self._calculate_overlap(chunk_text, other_text)
                    if overlap_ratio > 0.7:
                        is_subset = True
                        # 可能需要合并两个重叠块
                        if path != other_path and '.' in path and '.' in other_path:
                            # 如果是同一类的不同方法，选择路径更完整的版本
                            if path.split('.')[0] == other_path.split('.')[0]:
                                if len(path.split('.')) > len(other_path.split('.')):
                                    unique_chunks[j] = [chunk_text, path]
                        break
                except Exception as e:
                    # 出错时记录日志并继续，避免整个处理中断
                    logging.warning(f"Error calculating overlap: {e}")
                    continue
            
            if not is_subset:
                unique_chunks.append([chunk_text, path])
                chunk_hashes[content_hash] = len(unique_chunks) - 1
        
        # 第二步：识别类定义块和方法块，优先保持它们的连续性
        class_blocks = {}  # 类名 -> (块索引, 类定义)
        method_blocks = {}  # 类名 -> [方法块索引列表]
        special_methods = {}  # 类名 -> {特殊方法名: 索引}
        
        for i, (chunk_text, path) in enumerate(unique_chunks):
            # 检测类定义块
            if chunk_text.strip().startswith('class ') and ':' in chunk_text.split('\n')[0]:
                # 提取类名
                first_line = chunk_text.split('\n')[0]
                class_name = first_line.split('class ')[1].split('(')[0].split(':')[0].strip()
                class_blocks[class_name] = (i, first_line)
                
                # 如果路径中已有类名，使用路径中的类名（更准确）
                if '.' in path:
                    path_class = path.split('.')[-1]
                    if path_class != class_name:
                        class_blocks[path_class] = (i, first_line)
            
            # 检测方法块
            elif '.' in path:
                parts = path.split('.')
                method_name = parts[-1]
                
                # 确定类名
                if len(parts) > 1:
                    class_name = parts[-2]  # 直接父级为类名
                else:
                    class_name = parts[0]   # 单级路径时整个就是类名
                
                # 初始化集合
                if class_name not in method_blocks:
                    method_blocks[class_name] = []
                    special_methods[class_name] = {}
                
                method_blocks[class_name].append(i)
                
                # 标记特殊方法（如__init__）
                if method_name.startswith('__') and method_name.endswith('__'):
                    special_methods[class_name][method_name] = i
        
        # 第三步：智能合并处理
        result = []
        processed = set()  # 跟踪已处理的块
        current_chunk = None
        current_tokens = 0
        current_path = ""
        
        # 优先处理类及其方法
        for class_name, (class_idx, _) in class_blocks.items():
            if class_idx in processed:
                continue
            
            if class_name in method_blocks:
                # 尝试合并类和它的方法，优先处理特殊方法
                class_text, class_path = unique_chunks[class_idx]
                merged_chunk = class_text
                merged_tokens = num_tokens_from_string(class_text)
                merged_indices = [class_idx]
                
                # 优先处理__init__等特殊方法
                if class_name in special_methods:
                    for method_name, method_idx in special_methods[class_name].items():
                        if method_idx not in processed and method_idx != class_idx:
                            method_text, method_path = unique_chunks[method_idx]
                            method_tokens = num_tokens_from_string(method_text)
                            
                            # 对__init__方法给予更宽松的大小限制
                            size_limit = self.max_chunk_size
                            if method_name == '__init__':
                                size_limit = int(self.max_chunk_size * 1.1)  # 允许超出10%
                            
                            separator = "\n\n"
                            if merged_tokens + method_tokens + num_tokens_from_string(separator) <= size_limit:
                                merged_chunk += separator + method_text
                                merged_tokens += method_tokens + num_tokens_from_string(separator)
                                merged_indices.append(method_idx)
                
                # 处理其他普通方法
                for method_idx in method_blocks[class_name]:
                    if method_idx not in processed and method_idx not in merged_indices:
                        method_text, method_path = unique_chunks[method_idx]
                        method_tokens = num_tokens_from_string(method_text)
                        
                        # 尝试智能合并，避免只添加极小的片段
                        if method_tokens < self.min_chunk_size * 0.5:
                            # 对于非常小的方法，即使合并后稍微超过限制也可以接受
                            separator = "\n\n"
                            size_limit = int(self.max_chunk_size * 1.05)  # 允许超出5%
                        else:
                            separator = "\n\n"
                            size_limit = self.max_chunk_size
                        
                        if merged_tokens + method_tokens + num_tokens_from_string(separator) <= size_limit:
                            merged_chunk += separator + method_text
                            merged_tokens += method_tokens + num_tokens_from_string(separator)
                            merged_indices.append(method_idx)
                        elif "def " in method_text:
                            # 如果无法完整添加方法，尝试只添加方法签名
                            method_lines = method_text.split("\n")
                            method_sig = None
                            
                            # 查找方法签名行
                            for line in method_lines:
                                if "def " in line and ":" in line:
                                    method_sig = line + "\n    ..."
                                    break
                            
                            if method_sig:
                                sig_tokens = num_tokens_from_string(method_sig)
                                if merged_tokens + sig_tokens + num_tokens_from_string(separator) <= self.max_chunk_size:
                                    merged_chunk += separator + method_sig
                                    merged_tokens += sig_tokens + num_tokens_from_string(separator)
                                    # 注意：不标记为已处理，因为只用了方法签名
                
                # 如果合并了多个块，添加到结果
                if len(merged_indices) > 1:
                    result.append([merged_chunk, class_path])
                    processed.update(merged_indices)
                    continue
                elif len(merged_indices) == 1:
                    # 即使只有类定义本身，也标记为已处理
                    processed.add(class_idx)
        
        # 处理剩余的小块（相同代码）
        i = 0
        while i < len(unique_chunks):
            if i in processed:
                i += 1
                continue
            
            chunk_text, path = unique_chunks[i]
            chunk_tokens = num_tokens_from_string(chunk_text)
            
            # 如果当前块足够大，直接添加
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
                elif current_tokens + chunk_tokens + 2 <= self.max_chunk_size:  # +2 for newline
                    # 可以合并到当前累积块
                    # 添加分隔符，避免使用会破坏代码结构的注释
                    if current_path == path:
                        separator = "\n\n"
                    else:
                        # 智能分隔：同一类的不同部分用换行，不同类用Path注释
                        if '.' in current_path and '.' in path:
                            if current_path.split('.')[0] == path.split('.')[0]:
                                separator = "\n\n"
                            else:
                                separator = f"\n\n# Path: {path}\n"
                        else:
                            separator = f"\n\n# Path: {path}\n"
                    
                    current_chunk += separator + chunk_text
                    current_tokens += chunk_tokens + num_tokens_from_string(separator)
                else:
                    # 当前累积块已满，添加到结果并开始新块
                    result.append([current_chunk, current_path])
                    current_chunk = chunk_text
                    current_tokens = chunk_tokens
                    current_path = path
            
            processed.add(i)
            i += 1
        
        # 添加最后的累积块
        if current_chunk:
            result.append([current_chunk, current_path])
        
        return result

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


# 别名以与项目中其他解析器保持一致
CodeParser = RAGFlowCodeParser
