#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同温度下问题改写效果的脚本 - 每个温度测试5次
"""

import sys
import os
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from rag.prompts import full_question
from api import settings

settings.init_settings()


def test_question_refinement_with_temperature_multiple_runs():
    """
    测试不同温度下问题改写的效果 - 每个温度测试5次
    """

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(process)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 测试参数
    tenant_id = "6a2a5a8c00a611f0883a0242ac140006"  # 需要替换为实际的tenant_id
    llm_id = "Qwen/Qwen3-30B-A3B-FP8___LocalAI@LocalAI"  # 需要替换为实际的llm_id

    # 测试对话历史
    messages = [{'role': 'user', 'content': "\nRole: Expert in Rewriting Questions\n\nTask and steps:\n    1. Generate a full user question that would follow the conversation.\n    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is 2025-08-18. For example: 'yesterday' would be converted to 2025-08-17.\n\nRequirements & Restrictions:\n  - If the user's latest question is completely, don't do anything, just return the original question.\n  - DON'T generate anything except a refined question.\n  - Text generated MUST be in the same language of the original user's question.\n\n\n######################\n-Examples-\n######################\n\n# Example 1\n## Conversation\nUSER: What is the name of Donald Trump's father?\nASSISTANT:  Fred Trump.\nUSER: And his mother?\n###############\nOutput: What's the name of Donald Trump's mother?\n\n------------\n# Example 2\n## Conversation\nUSER: What is the name of Donald Trump's father?\nASSISTANT:  Fred Trump.\nUSER: And his mother?\nASSISTANT:  Mary Trump.\nUser: What's her full name?\n###############\nOutput: What's the full name of Donald Trump's mother Mary Trump?\n\n------------\n# Example 3\n## Conversation\nUSER: What's the weather today in London?\nASSISTANT:  Cloudy.\nUSER: What's about tomorrow in Rochester?\n###############\nOutput: What's the weather in Rochester on 2025-08-19?\n\n######################\n# Real Data\n## Conversation\nUSER: 解释下新手礼包事故\nASSISTANT: 新手礼包事故主要涉及两个核心问题：**功能逻辑缺陷导致用户误领卡券**，以及**报表数据错误**，最终影响了活动的公平性和运营分析的准确性。以下是详细解释：\n\n---\n\n### 🔍 一、事故根本原因：领取条件校验逻辑存在严重缺陷\n\n#### 📌 问题描述：\n在六周年新手礼包活动中，系统未能正确校验用户是否满足卡券领取条件，导致**非目标用户也成功领取了卡券**，破坏了活动的精准投放和公平性。\n\n#### 🛠️ 技术细节：\n- 新增了一种“二级时间条件”（如：注册后7天内完成首次充值），这类条件不能独立验证，而是要**动态限制其他条件的生效范围**。\n- 原始代码中，该时间条件被解析后存储在实例变量中，后续验证其他条件时会读取这个变量。\n- **但问题出在：** 如果“二级时间条件”没有放在所有条件的**第一个**，系统就无法正确获取该时间范围，导致后续条件校验失效。\n  - 举例：用户注册时间在活动开始后第10天，但配置的条件是“注册7天内完成充值”，本应不满足。但由于时间条件未被正确识别，系统误判为满足，允许领取。\n\n> ✅ **修复方案**：在验证任何条件前，先扫描所有条件，统一提取并保存“二级时间条件”到一个共享的元变量中，确保无论顺序如何，都能正确限制范围。##2$$\n\n---\n\n### 📊 二、报表数据错误：数据源缺失 + 逻辑错误双重打击\n\n#### 📌 问题表现：\n- 活动期间的报表数据显示异常，无法准确反映真实用户参与情况。\n- 活动结束后，部分数据无法修复，影响了后续运营决策。\n\n#### 🛠️ 原因分析：\n1. **前端上报接口未正常工作**：\n   - 前端配置错误，导致关键事件（如“卡券领取成功”）没有上报。\n   - 事件追踪系统依赖上报数据生成报表，数据缺失直接导致报表为空或错误。\n\n2. **后端逻辑错误叠加**：\n   - 由于领取条件校验失败，大量非目标用户领取了卡券，但这些数据仍被计入报表。\n   - 报表系统没有做去重或异常过滤，导致数据严重失真。\n\n> 💡 **关键教训**：上线前未在真实环境测试前端事件上报，也未对复杂条件逻辑进行充分自测，是事故扩大的主因。##2$$\n\n---\n\n### ✅ 三、事故处理与改进措施\n\n| 问题 | 解决方案 |\n|------|----------|\n| 卡券误发 | 活动第三天上线修复补丁，阻止非目标用户继续领取 ##2$$ |\n| 报表数据错误 | 活动期间手动导出原始数据，活动结束后通过修复后的报表系统重新计算并修正数据 |\n| 防止再发生 | 强化需求评估流程，复杂功能需组长评审；增加单元测试和线上验证环节 |\n\n---\n\n### 📌 总结：事故本质是“设计缺陷 + 测试缺失”叠加\n\n> 新手礼包事故并非单一技术故障，而是：\n> - **需求设计不合理**：强行引入“二级时间条件”打破原有“独立条件并集”的逻辑；\n> - **开发验证不足**：未在真实场景下测试条件顺序、数据上报等关键路径；\n> - **缺乏兜底机制**：报表系统未设置异常检测与数据校验。\n\n> ⚠️ 启示：复杂功能必须“先设计、再开发、后验证”，避免“边做边改”带来连锁风险。##2$$\n\n--- \n\n📌 **建议**：如你正在运营类似活动，务必：\n- 严格测试条件组合顺序；\n- 上线前模拟真实用户行为验证事件上报；\n- 建立报表数据校验机制，发现异常及时预警。\nUSER: 结合图片说明\n###############\n    "}]

    # 测试不同的温度值
    temperatures = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    runs_per_temperature = 5  # 每个温度测试5次

    print("=" * 100)
    print("测试不同温度下问题改写效果 - 每个温度测试5次")
    print("=" * 100)
    print(f"原始问题: {messages[-1]['content']}")
    print(f"对话历史: {[msg['content'] for msg in messages[:-1]]}")
    print(f"每个温度测试次数: {runs_per_temperature}")
    print()

    # 创建LLM模型实例
    try:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
        print(f"成功创建LLM模型: {chat_mdl.llm_name}")
        print()
    except Exception as e:
        print(f"创建LLM模型失败: {e}")
        return

    # 存储所有结果
    all_results = defaultdict(list)

    # 测试每个温度值
    for temp in temperatures:
        print(f"温度: {temp}")
        print("-" * 60)

        for run in range(runs_per_temperature):
            print(f"  第 {run + 1} 次测试:")

            try:

                # 使用指定温度调用模型
                gen_conf = {"temperature": temp}
                logging.info(f"使用温度 {temp} 调用模型 - 第 {run + 1} 次")

                start_time = time.time()
                result = chat_mdl.chat(None,history=messages, gen_conf=gen_conf)
                end_time = time.time()

                # 清理结果
                import re
                result = re.sub(r"<think>.*</think>", "", result, flags=re.DOTALL)

                if result.find("**ERROR**") >= 0:
                    print(f"    错误: {result}")
                    all_results[temp].append({"error": result, "time": end_time - start_time})
                else:
                    print(f"    结果: {result}")
                    print(f"    耗时: {end_time - start_time:.2f}秒")
                    all_results[temp].append({"result": result, "time": end_time - start_time})

                # 添加延迟避免API限制
                time.sleep(1)

            except Exception as e:
                print(f"    错误: {e}")
                all_results[temp].append({"error": str(e), "time": 0})

        print()

    # 分析结果
    analyze_results(all_results, temperatures)


def test_multiple_questions_with_temperature_multiple_runs():
    """
    测试多个问题在不同温度下的改写效果 - 每个温度测试5次
    """

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(process)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 测试参数
    tenant_id = "your_tenant_id"  # 需要替换为实际的tenant_id
    llm_id = "your_llm_id"  # 需要替换为实际的llm_id

    # 测试多个对话场景
    test_cases = [
        {
            "name": "卡券职能投放场景",
            "messages": [
                {"role": "user", "content": "解释下卡券职能投放"},
                {"role": "assistant",
                 "content": "卡券职能投放是指通过系统自动或手动方式将优惠券、礼品卡等发放给指定用户群体的过程。"},
                {"role": "user", "content": "解释下新手礼包事故"},
                {"role": "assistant",
                 "content": "新手礼包事故是指在发放新手礼包过程中出现的异常情况，如重复发放、发放失败等。"},
                {"role": "user", "content": "结合图片进行说明"}
            ]
        },
        {
            "name": "技术文档场景",
            "messages": [
                {"role": "user", "content": "什么是API？"},
                {"role": "assistant", "content": "API是应用程序编程接口的缩写，是不同软件应用程序之间进行通信的接口。"},
                {"role": "user", "content": "RESTful API有什么特点？"},
                {"role": "assistant", "content": "RESTful API具有无状态、可缓存、统一接口等特点。"},
                {"role": "user", "content": "如何设计一个好的API？"}
            ]
        },
        {
            "name": "简单对话场景",
            "messages": [
                {"role": "user", "content": "今天天气怎么样？"},
                {"role": "assistant", "content": "今天天气晴朗，温度25度。"},
                {"role": "user", "content": "明天呢？"}
            ]
        }
    ]

    # 测试温度值
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    runs_per_temperature = 5  # 每个温度测试5次

    try:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
        print(f"成功创建LLM模型: {chat_mdl.llm_name}")
        print()
    except Exception as e:
        print(f"创建LLM模型失败: {e}")
        return

    for test_case in test_cases:
        print("=" * 100)
        print(f"测试场景: {test_case['name']}")
        print("=" * 100)

        messages = test_case["messages"]
        original_question = messages[-1]["content"]
        print(f"原始问题: {original_question}")
        print(f"对话历史: {[msg['content'] for msg in messages[:-1]]}")
        print(f"每个温度测试次数: {runs_per_temperature}")
        print()

        # 存储当前场景的结果
        scene_results = defaultdict(list)

        for temp in temperatures:
            print(f"温度 {temp}:")

            for run in range(runs_per_temperature):
                print(f"  第 {run + 1} 次:")

                try:
                    # 构建对话历史
                    conv = []
                    for m in messages:
                        if m["role"] in ["user", "assistant"]:
                            conv.append("{}: {}".format(m["role"].upper(), m["content"]))
                    conv = "\n".join(conv)

                    # 构建prompt
                    today = datetime.now().date().isoformat()
                    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
                    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()

                    prompt = f"""
Role: Expert in Rewriting Questions

Task and steps:
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {today}. For example: 'yesterday' would be converted to {yesterday}.

Requirements & Restrictions:
  - If the user's latest question is completely, don't do anything, just return the original question.
  - DON'T generate anything except a refined question.
  - Text generated MUST be in the same language of the original user's question.

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {tomorrow}?

######################
# Real Data
## Conversation
{conv}
###############
"""

                    # 调用模型
                    gen_conf = {"temperature": temp}
                    start_time = time.time()
                    result = chat_mdl.chat(prompt, [{"role": "user", "content": "Output: "}], gen_conf)
                    end_time = time.time()

                    # 清理结果
                    import re
                    result = re.sub(r"<think>.*</think>", "", result, flags=re.DOTALL)

                    if result.find("**ERROR**") >= 0:
                        print(f"    错误: {result}")
                        scene_results[temp].append({"error": result, "time": end_time - start_time})
                    else:
                        print(f"    {result}")
                        scene_results[temp].append({"result": result, "time": end_time - start_time})

                    # 添加延迟避免API限制
                    time.sleep(1)

                except Exception as e:
                    print(f"    错误: {e}")
                    scene_results[temp].append({"error": str(e), "time": 0})

            print()

        # 分析当前场景的结果
        print("当前场景结果分析:")
        analyze_results(scene_results, temperatures)
        print()


def analyze_results(results, temperatures):
    """
    分析测试结果
    """
    print("结果分析:")
    print("-" * 40)

    for temp in temperatures:
        if temp not in results:
            continue

        temp_results = results[temp]
        successful_runs = [r for r in temp_results if "result" in r]
        error_runs = [r for r in temp_results if "error" in r]

        print(f"温度 {temp}:")
        print(f"  成功次数: {len(successful_runs)}/{len(temp_results)}")
        print(f"  错误次数: {len(error_runs)}/{len(temp_results)}")

        if successful_runs:
            # 计算平均响应时间
            avg_time = sum(r["time"] for r in successful_runs) / len(successful_runs)
            print(f"  平均响应时间: {avg_time:.2f}秒")

            # 显示所有成功的结果
            print(f"  改写结果:")
            for i, result in enumerate(successful_runs, 1):
                print(f"    {i}. {result['result']}")

            # 分析结果的一致性
            unique_results = set(r["result"] for r in successful_runs)
            consistency = len(unique_results) / len(successful_runs) if successful_runs else 0
            print(f"  结果一致性: {consistency:.2%} (1.0表示完全相同，0.0表示完全不同)")

        if error_runs:
            print(f"  错误信息:")
            for i, error in enumerate(error_runs, 1):
                print(f"    {i}. {error['error']}")

        print()


def generate_summary_report():
    """
    生成总结报告
    """
    print("=" * 100)
    print("温度参数对问题改写效果的影响总结")
    print("=" * 100)

    print("""
温度参数说明:
- 0.1: 非常保守，输出最确定性的结果，多次运行结果应该高度一致
- 0.2: 相对保守，保持一定的创造性
- 0.3: 平衡模式，在确定性和创造性之间平衡
- 0.5: 中等创造性，允许适度变化
- 0.7: 相对创造性，允许更多变化
- 0.9: 高创造性，输出更多样化
- 1.0: 最大创造性，输出最不可预测

观察要点:
1. 低温度(0.1-0.3): 
   - 改写结果应该更一致
   - 结果一致性应该接近1.0
   - 更接近原始问题的语义

2. 中温度(0.5): 
   - 在保持相关性的同时增加一些创造性
   - 结果一致性应该在0.6-0.8之间

3. 高温度(0.7-1.0): 
   - 改写结果更多样化
   - 结果一致性应该较低(0.3-0.6)
   - 可能产生更丰富的表达

4. 响应时间:
   - 不同温度下的响应时间应该相对稳定
   - 如果某个温度响应时间异常，可能是模型负载问题

建议:
- 对于需要准确性和一致性的场景，使用低温度(0.1-0.3)
- 对于需要适度创造性的场景，使用中温度(0.5)
- 对于需要多样性的场景，使用中高温度(0.7-0.9)
- 避免使用过高温度(>0.9)，可能导致不相关或不一致的结果
- 根据实际应用场景选择合适的一致性阈值
""")


if __name__ == "__main__":
    print("问题改写温度测试脚本 - 每个温度测试5次")
    print("请确保已正确配置tenant_id和llm_id")
    print()

    # 运行测试
    try:
        # 测试单个问题
        test_question_refinement_with_temperature_multiple_runs()

        # 测试多个问题
        # test_multiple_questions_with_temperature_multiple_runs()

        # 生成总结报告
        generate_summary_report()

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")