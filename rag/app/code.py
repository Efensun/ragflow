import logging
import os
import re
from io import BytesIO

from rag.nlp import rag_tokenizer, tokenize_chunks
from rag.utils import num_tokens_from_string
from deepdoc.parser import TxtParser
from deepdoc.parser.code_parser import CodeParser


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Python", callback=None, **kwargs):
    """
    Parse and chunk code files into manageable segments.

    Args:
        filename: File name or path
        binary: Binary content of the file
        from_page: Starting page (not used for code files)
        to_page: Ending page (not used for code files)
        lang: Programming language
        callback: Callback function for progress reporting
        **kwargs: Additional arguments

    Returns:
        List of code chunks with their metadata
    """
    callback(0.1, "Start to parse code.")

    # 准备文档信息
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    # 获取解析配置
    parser_config = kwargs.get(
        "parser_config", {
            "chunk_token_num": 512,
            "delimiter": "\n"
        })

    max_chunk_size = int(parser_config.get("chunk_token_num", 512))

    # 检测文件扩展名
    if isinstance(filename, str):
        file_extension = os.path.splitext(filename)[1].lower()
    else:
        # 如果输入是二进制内容，尝试从kwargs中获取扩展名或默认为.py
        file_extension = kwargs.get("file_extension", ".py")

    # 根据文件类型选择合适的解析器
    code_parser = None
    try:
        # 目前只支持Python，未来可以添加更多语言支持
        if file_extension in ['.py', '.pyx', '.pyi', '.pyw']:
            code_parser = CodeParser()
        else:
            # 对于不支持的代码类型，使用TxtParser
            code_parser = TxtParser()
    except ImportError as e:
        logging.warning(f"Code parser initialization failed: {e}")
        logging.warning("Falling back to text parser")
        code_parser = TxtParser()

    callback(0.3, "Parser initialized.")

    # 解析代码文件
    try:
        sections = code_parser(
            filename,
            binary,
            max_chunk_size=max_chunk_size
        )
        callback(0.8, "Parsing completed.")
    except Exception as e:
        logging.exception(f"Error parsing code file: {e}")
        callback(0.8, f"Error parsing code file: {e}")
        return []

    # 如果只需要返回代码段，则直接返回
    if kwargs.get("section_only", False):
        return sections

    # 否则标记化生成嵌入向量
    is_english = lang.lower() == "english"
    # 这里进行修改：确保sections符合tokenize_chunks预期的格式
    # CodeParser返回的是[chunk_text, path]的列表，需要转换为字符串列表
    modified_sections = []
    for section in sections:
        if isinstance(section, list) and len(section) == 2:
            # 如果section是[text, path]格式，就将text和path合并
            text, path = section
            if path:
                modified_sections.append(f"{text} // Path: {path}")
            else:
                modified_sections.append(text)
        else:
            # 其他情况保持原样
            modified_sections.append(section)

    result = tokenize_chunks(modified_sections, doc, is_english)

    callback(1.0, "Tokenization completed.")
    return result