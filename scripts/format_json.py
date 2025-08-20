import json


def convert_to_openai_format(source_payload: dict) -> dict:
    """
    将包含 'system' 和 'history' 键的自定义格式，
    转换为包含 'messages' 数组的标准 OpenAI API 格式。

    Args:
        source_payload (dict): 原始格式的字典数据。

    Returns:
        dict: 转换后符合 OpenAI 标准的字典数据。
    """
    # 初始化一个新的列表，用于存放所有消息
    messages = []

    # 1. 处理 'system' 消息
    # 检查 'system' 键是否存在且有内容
    if 'system' in source_payload and source_payload['system']:
        system_message = {
            "role": "system",
            "content": source_payload['system']
        }
        messages.append(system_message)

    # 2. 处理 'history' 列表
    # 检查 'history' 键是否存在且是一个列表
    if 'history' in source_payload and isinstance(source_payload.get('history'), list):
        # 使用 extend 将 history 列表中的所有元素追加到 messages 列表
        messages.extend(source_payload['history'])

    # 3. 构建最终的目标 payload
    # 首先，复制原始 payload 中除了 'system' 和 'history' 之外的所有键值对
    # 这样可以保留如 'model', 'temperature', 'max_tokens' 等其他参数
    target_payload = {
        key: value
        for key, value in source_payload.items()
        if key not in ['system', 'history']
    }

    # 4. 将格式化好的 messages 列表添加到新的 payload 中
    target_payload['messages'] = messages

    return target_payload


# --- 主程序入口，用于演示 ---
if __name__ == "__main__":
    # 1. 定义一个您提出的非标准格式的数据
    import json

    data = {}
    with open('test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 调用转换函数
    correct_format_data = convert_to_openai_format(data)

    # 3. 打印结果进行对比
    print("--- 原始格式 (非标准) ---")
    # 使用 json.dumps 美化打印输出，ensure_ascii=False 确保中文正常显示
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\n" + "=" * 40 + "\n")

    print("--- 转换后的格式 (OpenAI 标准) ---")
    print(json.dumps(correct_format_data, indent=2, ensure_ascii=False))