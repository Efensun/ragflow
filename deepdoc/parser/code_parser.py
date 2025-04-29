import logging
import os
import re
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from typing import List, Dict, Tuple, Generator, Any, Callable, Optional, Union

from deepdoc.parser.utils import get_text
from rag.utils import num_tokens_from_string


class CodeSplitter:
    """代码分割器，使用AST解析器分割代码。

    基于llama_index的实现，感谢Kevin Lu/SweepAI提供的优雅的代码分割解决方案。
    https://docs.sweep.dev/blogs/chunking-2m-files
    """

    def __init__(
            self,
            language: str = 'python',
            chunk_lines: int = 40,
            chunk_lines_overlap: int = 15,
            max_chunk_size: int = 512,
            parser: Any = None,
    ):
        """初始化代码分割器

        Args:
            language: 要解析的编程语言
            chunk_lines: 每个分块包含的行数
            chunk_lines_overlap: 每个分块与相邻分块的重叠行数
            max_chunk_size: 每个分块的最大token数量
            parser: 可选的tree-sitter解析器对象
        """
        self.language = language
        self.chunk_lines = chunk_lines
        self.chunk_lines_overlap = chunk_lines_overlap
        self.max_chunk_size = max_chunk_size

        # 初始化解析器
        if parser is None:
            try:
                import tree_sitter_language_pack
                parser = tree_sitter_language_pack.get_parser(language)
            except ImportError:
                raise ImportError(
                    "请安装tree_sitter_language_pack来使用CodeSplitter，"
                    "或者传入一个解析器对象。"
                )
            except Exception:
                print(
                    f"无法获取语言 {language} 的解析器。请检查 "
                    "https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages "
                    "获取有效语言列表。"
                )
                raise

        if not isinstance(parser, Parser):
            raise ValueError("解析器必须是tree-sitter Parser对象")

        self._parser = parser

    def __call__(self, filename, binary=None, max_chunk_size=512, **kwargs):
        """解析代码文件并进行分块

        Args:
            filename: 文件名或路径
            binary: 文件的二进制内容
            max_chunk_size: 最大token分块大小
            **kwargs: 额外参数

        Returns:
            代码分块的列表
        """
        self.max_chunk_size = max_chunk_size
        code_text = get_text(filename, binary)
        file_path = filename if isinstance(filename, str) else kwargs.get("file_path", "unknown")

        try:
            chunks = self.split_text(code_text)
            # 将分块转换成所需的格式 [文本, 路径]
            return [[chunk, file_path] for chunk in chunks]
        except ValueError as e:
            logging.warning(f"解析代码时出错: {e}")
            # 使用简单的基于行的分块作为后备方案
            return self._fallback_parsing(code_text, file_path)

    def _chunk_node(self, node: Any, text: str) -> List[str]:
        """递归地将节点分成更小的块

        Args:
            node: 要分块的AST节点
            text: 原始源代码文本

        Returns:
            代码块列表
        """
        new_chunks = []
        current_chunk = ""

        for child in node.children:
            child_text = text[child.start_byte:child.end_byte]
            child_tokens = num_tokens_from_string(child_text)

            # 如果子节点太大，递归处理
            if child_tokens > self.max_chunk_size:
                if current_chunk:
                    new_chunks.append(current_chunk)
                    current_chunk = ""
                new_chunks.extend(self._chunk_node(child, text))
            # 如果当前累积块加上子节点会过大，开始新块
            elif current_chunk and num_tokens_from_string(current_chunk + child_text) > self.max_chunk_size:
                new_chunks.append(current_chunk)
                current_chunk = child_text
            else:
                current_chunk += child_text

        if current_chunk:
            new_chunks.append(current_chunk)

        return new_chunks

    def split_text(self, text: str) -> List[str]:
        """使用AST解析器将输入代码分成块

        该方法解析输入代码为AST，然后在保留语法结构的同时进行分块。
        它处理错误情况并确保代码可以被正确解析。

        Args:
            text: 要分割的源代码文本

        Returns:
            代码块列表

        Raises:
            ValueError: 如果指定语言的代码无法解析
        """
        tree = self._parser.parse(bytes(text, "utf-8"))

        if not tree.root_node.children or tree.root_node.children[0].type != "ERROR":
            chunks = [chunk.strip() for chunk in self._chunk_node(tree.root_node, text)]

            # 确保每个块不超过最大token大小
            valid_chunks = []
            for chunk in chunks:
                token_count = num_tokens_from_string(chunk)
                if token_count > 0 and token_count <= self.max_chunk_size:
                    valid_chunks.append(chunk)
                elif token_count > self.max_chunk_size:
                    # 简单分割过大的块
                    lines = chunk.split('\n')
                    current_lines = []
                    current_tokens = 0

                    for line in lines:
                        line_tokens = num_tokens_from_string(line)
                        if current_tokens + line_tokens <= self.max_chunk_size:
                            current_lines.append(line)
                            current_tokens += line_tokens
                        else:
                            if current_lines:
                                valid_chunks.append('\n'.join(current_lines))
                            current_lines = [line]
                            current_tokens = line_tokens

                    if current_lines:
                        valid_chunks.append('\n'.join(current_lines))

            return valid_chunks
        else:
            raise ValueError(f"无法使用语言 {self.language} 解析代码")

    def _fallback_parsing(self, code_text: str, file_path: str) -> List[List[Any]]:
        """当tree-sitter解析失败时的备用解析方法

        Args:
            code_text: 代码文本
            file_path: 文件路径

        Returns:
            代码块列表
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
                    chunks.append(['\n'.join(current_chunk), file_path])
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
                        chunks.append([' '.join(temp_line), file_path])
                        temp_line = [word]
                        temp_tokens = word_tokens

                if temp_line:
                    chunks.append([' '.join(temp_line), file_path])

            # 常规情况下添加行到当前块
            elif current_tokens + line_tokens <= self.max_chunk_size:
                current_chunk.append(line)
                current_tokens += line_tokens
            else:
                chunks.append(['\n'.join(current_chunk), file_path])
                current_chunk = [line]
                current_tokens = line_tokens

        # 添加最后的块
        if current_chunk:
            chunks.append(['\n'.join(current_chunk), file_path])

        return chunks


# 别名以与项目中其他解析器保持一致
CodeParser = CodeSplitter