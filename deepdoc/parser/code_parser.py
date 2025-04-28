import logging
import os
import re
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from deepdoc.parser.utils import get_text
from rag.utils import num_tokens_from_string


class RAGFlowCodeParser:
    """Parser for code files with tree-sitter support"""
    
    def __init__(self):
        # 初始化Python解析器和语言
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser()
        self.parser = Parser(self.PY_LANGUAGE)
        self.max_chunk_size = 512
    
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
        
        # 检测文件类型
        file_extension = os.path.splitext(filename)[1].lower() if isinstance(filename, str) else '.py'
        
        # 目前仅支持Python
        if file_extension == '.py':
            return self._parse_python_code(code_text)
        else:
            # 对于不支持的文件类型，使用简单分割
            return self._fallback_parsing(code_text)
    
    def _parse_python_code(self, code_text):
        """Parse Python code using tree-sitter

        Args:
            code_text: Python code text

        Returns:
            List of code chunks with their metadata
        """
        code_bytes = code_text.encode('utf-8')
        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node
        
        # 获取代码块
        chunks = self._extract_code_chunks(root_node, code_bytes)
        logging.info(f"code parser found {len(chunks)} chunks")
        # 如果没有找到有意义的代码块，则使用备用解析
        if not chunks:
            return self._fallback_parsing(code_text)
        
        return chunks
    
    def _extract_code_chunks(self, node, code_bytes, path=""):
        """Extract code chunks from the syntax tree
        
        Args:
            node: Tree-sitter node
            code_bytes: Code as bytes
            path: Current code path (class.method etc.)
            
        Returns:
            List of code chunks
        """
        chunks = []
        
        for child in node.children:
            current_path = path
            
            if child.type in ('function_definition', 'class_definition'):
                # 获取名称节点
                name_node = None
                
                for field_name in ['name', 'identifier']:
                    try:
                        name_node = child.child_by_field_name(field_name)
                        if name_node:
                            break
                    except:
                        pass
                
                if not name_node:
                    for c in child.children:
                        if c.type == 'identifier':
                            name_node = c
                            break
                
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode()
                    current_path = f"{path}.{name}" if path else name
                
                # 提取代码块
                chunk_text = code_bytes[child.start_byte:child.end_byte].decode()
                
                # 检查代码块大小
                if num_tokens_from_string(chunk_text) <= self.max_chunk_size:
                    chunks.append([chunk_text, current_path])
                else:
                    # 如果代码块太大，我们需要使用压缩的代码块表示
                    if child.type == 'class_definition':
                        # 对于类，保留类定义和方法签名
                        chunks.append([self._collapse_class(child, code_bytes), current_path])
                    elif child.type == 'function_definition':
                        # 对于函数，保留函数签名和折叠函数体
                        chunks.append([self._collapse_function(child, code_bytes), current_path])
            
            # 递归处理所有子节点以获取更多代码块
            sub_chunks = self._extract_code_chunks(
                child, code_bytes, current_path if child.type in ('function_definition', 'class_definition') else path
            )
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _collapse_class(self, node, code_bytes):
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
        
        # 获取类定义和类体开始部分
        class_def = code_bytes[node.start_byte:body_node.start_byte].decode()
        body_start = code_bytes[body_node.start_byte:body_node.start_byte+1].decode()
        
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
                    # 获取方法签名
                    method_sig = code_bytes[child.start_byte:func_body.start_byte].decode().strip()
                    method_signatures.append("    " + method_sig + ":\n        ...")
        
        # 构建折叠后的类定义
        collapsed_class = f"{class_def}{body_start}\n"
        collapsed_class += "\n".join(method_signatures)
        collapsed_class += "\n}"
        
        return collapsed_class
    
    def _collapse_function(self, node, code_bytes):
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
        
        # 获取函数签名和函数体的第一行
        func_sig = code_bytes[node.start_byte:body_node.start_byte].decode()
        
        # 构建折叠后的函数定义
        collapsed_func = f"{func_sig}:\n    ..."
        
        return collapsed_func
    
    def _fallback_parsing(self, code_text):
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
        
        return chunks


# 别名以与项目中其他解析器保持一致
CodeParser = RAGFlowCodeParser
