#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同温度下问题改写效果的脚本 - 每个温度测试5次
"""

import sys
import os
import logging
import time
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
    llm_id = "gemini-2.5-flash___OpenAI-API@OpenAI-API-Compatible"  # 需要替换为实际的llm_id

    system = """
    \nRole: You are an expert question rewriter with deep expertise in natural language understanding and contextual analysis.\n\nCore Mission:\nTransform incomplete, ambiguous, or contextual user questions into standalone, complete, and precise questions that can be understood independently.\n\nChain of Thought Process:\n<thinking>\n1. Context Analysis: Carefully analyze the entire conversation history to understand the main topic and discussion flow\n2. Reference Resolution: Identify pronouns, ellipses, and implicit references in the question, determining what specific entities they refer to\n3. Temporal Conversion: Convert relative time expressions to absolute dates (current date: 2025-08-27)\n4. Completeness Check: Ensure the rewritten question contains all necessary information for independent understanding\n5. Semantic Preservation: Verify that the rewritten question maintains the exact original intent and meaning\n</thinking>\n\nSelf-Consistency Verification:\nBefore providing your final answer, perform these critical checks:\n- Does the rewritten question preserve the core intent of the original question?\n- Does it contain sufficient contextual information to be understood independently?\n- Are temporal expressions correctly converted to absolute dates?\n- Is the language and tone consistent with the original question?\n\nQuality Commitment:\nYour accurate rewriting directly impacts the user\'s ability to receive precise answers. Apply your professional expertise to ensure every rewrite is precisely crafted and contextually complete.\n\nTask Requirements:\n1. Generate a complete user question that logically follows the conversation context.\n2. Convert relative dates to absolute dates based on current date: 2025-08-27\n   - \'yesterday\' becomes 2025-08-26\n   - \'today\' remains 2025-08-27\n   - \'tomorrow\' becomes 2025-08-28\n\nCritical Guidelines:\n- If the user\'s latest question is already complete and clear, return the original question unchanged\n- Output ONLY the refined question without any additional explanation or commentary\n- Generated text MUST be in the same language as the original user\'s question\n\nExamples with Chain of Thought:\n\n# Example 1: Pronoun Reference Resolution\n## Conversation\nUSER: What is the name of Donald Trump\'s father?\nASSISTANT: Fred Trump.\nUSER: And his mother?\n\n<thinking>\n1. Context: Discussion about Trump\'s family members\n2. Reference: "his" refers to Donald Trump\n3. Ellipsis: asking about mother\'s name\n4. Complete question: What\'s the name of Donald Trump\'s mother?\n</thinking>\n\nOutput: What\'s the name of Donald Trump\'s mother?\n\n------------\n# Example 2: Chain Reference Resolution\n## Conversation\nUSER: What is the name of Donald Trump\'s father?\nASSISTANT: Fred Trump.\nUSER: And his mother?\nASSISTANT: Mary Trump.\nUSER: What\'s her full name?\n\n<thinking>\n1. Context: First asked about father, then mother, now asking for mother\'s full name\n2. Reference: "her" refers to Mary Trump (Trump\'s mother)\n3. Complete question: Need to specify it\'s about Trump\'s mother Mary Trump\'s full name\n</thinking>\n\nOutput: What\'s the full name of Donald Trump\'s mother Mary Trump?\n\n------------\n# Example 3: Temporal and Location Conversion\n## Conversation\nUSER: What\'s the weather today in London?\nASSISTANT: Cloudy.\nUSER: What about tomorrow in Rochester?\n\n<thinking>\n1. Context: Weather inquiry\n2. Temporal conversion: "tomorrow" needs to be converted to specific date 2025-08-28\n3. Location shift: from London to Rochester\n4. Complete question: Need to include specific date and location\n</thinking>\n\nOutput: What\'s the weather in Rochester on 2025-08-28?\n\n######################\n# Actual Task\n## Conversation\nUSER: 币种评论是做什么的\nASSISTANT: 您好！很高兴能为您详细解释“币种评论”是做什么的。😊\n\n“币种评论”是平台为了**丰富币种信息的全面性、提高用户活跃度和粘性**而推出的一个核心功能模块。它旨在将复杂的技术知识转化为用户易于理解和应用的实用信息，并提供一个让用户可以**查看并参与讨论**的平台。##2$$\n\n### 🌟 功能定位与目标\n\n从不同的角度来看，“币种评论”有以下几个主要目标：\n\n*   **业务层面**：\n    *   **丰富信息**：通过接入外部渠道和鼓励用户原创内容，提升币种信息的全面性。##2$$\n    *   **提高活跃度与粘性**：用户可以查看并参与讨论，从而增加在平台的互动。##2$$\n    *   **内容价值**：有价值的评论后续经过人工审核，甚至可以支持推送，扩大影响力。##2$$\n\n*   **用户层面**：\n    *   **了解动态**：用户可以通过评论了解币种最新动态，辅助交易决策。##2$$\n    *   **分享见解**：用户可以发表对币种的看法，传播自己的见解，并期望获得反馈和认同。##2$$\n\n*   **系统层面**：\n    *   **组件化**：将评论功能设计成可复用的组件，方便未来低成本地接入到其他业务模块，确保交互一致性，降低维护成本。##2$$\n\n### 🛠️ 核心功能概览\n\n“币种评论”模块包含了多个核心功能，共同构建了一个完整的互动体验：\n\n1.  **评论区类别**：支持**多向评论**，用户不仅可以发表自己的看法，还能与其他用户交流互动。##2$$\n2.  **应用业务**：主要应用于**币种资料页面**，但作为组件化功能，后续可轻松接入其他业务场景。##2$$\n3.  **包含模块**：\n    *   **账号信息展示**：显示用户的账户名、用户名和头像。\n    *   **评论输入**：提供评论和回复的入口。\n    *   **评论展示**：显示评论内容、时间、点赞/反对数等。\n    *   **评论排序**：支持**热门（Hot）**、**最新（New）**、**互动（Top）**等多种排序方式，确保热门内容不会过快沉下去，同时兼顾新内容的曝光。##2$$ ##1$$\n    *   **评论互动**：支持点赞（赞同）、点踩（反对）、举报等功能。##2$$\n    *   **消息触达**：通过站内信和推送（Push）通知用户相关的互动消息。##2$$\n4.  **风险控制**：\n    *   **关键词过滤**：通过 Admin 设置的关键词和接入的第三方 AI 机器人进行初步筛查和全面筛查。##2$$\n    *   **安全性**：防止 XSS 攻击、SQL 注入等安全漏洞。##2$$\n5.  **Admin 管理和审核**：\n    *   管理员可以管理过滤词、评论内容、待审核内容、评论账户数据，并对用户进行禁言操作。##2$$\n    *   举报人数达到阈值（例如超过10次）的评论会进入人工审核。##2$$\n\n### 📊 评论处理流程与数据结构\n\n为了实现这些功能，系统在后端设计上非常严谨：\n\n*   **数据模型**：评论和回复被设计在同一个 `Comment` 模型中，因为它们在排序、状态、投票和审核等操作上具有一致性。##3$$\n    *   ![Comment Model Fields](https://t9008230771.p.clickup-attachments.com/t9008230771/dd6f82ec-29be-4796-8f46-93ad74bf7c10/image.png)\n    *   这张图片展示了AI翻译按钮的展示逻辑，当评论本身的语言与社区语区不一致时，会显示AI翻译按钮。##2$$\n*   **评论状态**：评论会经历从 `CREATED`（刚创建，仅自己可见）到 `PUBLISHED`（审核通过，公开展示）等状态流转，确保内容合规。##3$$\n    *   ![Key Status Transitions](http://120.77.38.66:8009/images/b11e51cb651725433d85029f4e968a5a-582497.svg)\n    *   这张图描绘了评论从创建到发布、禁用或删除的关键状态流转，体现了内容审核的严谨性。##3$$\n*   **缓存设计**：为了支持高并发，系统采用了评论缓存、评论列表缓存和用户待审核评论缓存等机制。##3$$\n    *   ![Comment Cache Flow](http://120.77.38.66:8009/images/5ea402f649ab1075e820240d0276efc0-337147.svg)\n    *   这个流程图展示了评论经过初筛后如何写入数据库和缓存，确保高效处理。##3$$\n\n### ⚖️ 排序算法\n\n币种评论的排序是其核心功能之一。平台参考了 Reddit 和 Hack News 等成熟社交平台的算法，并结合自身需求进行了调整：\n\n*   **Hot 排序**：采用了类似 Reddit 故事排序的算法，平衡了用户互动（赞成-反对）和时间衰减的影响，确保热门内容和新内容都能得到曝光。##1$$ ##3$$\n    *   ![Reddit Story Sorting Formula](https://t9008230771.p.clickup-attachments.com/t9008230771/902bad48-3922-44cd-a358-021dfb5ca084/image.png)\n    *   这个公式是 Reddit 故事排序的核心，它通过调整参数来降低提交时间对评论排名的影响。##1$$\n*   **New 排序**：按发布时间倒序排列。##2$$\n*   **Top 排序**：主要基于评论互动（赞成-反对）的得分，得分高的排名靠前。##2$$\n*   **评论数量的影响**：在调研中发现，Reddit 和 Hack News 的排序算法并未将评论数量纳入考虑，而平台可能根据实际情况决定是否将其作为排序参数。##1$$\n\n### ✨ 新增打赏功能\n\n为了进一步激励优质内容创作，币种评论还计划新增**打赏功能**：\n\n*   **打赏币种**：目前仅支持 CET。##6$$\n*   **额度限制**：有每日打赏限额、现货可用余额和24小时提现额度限制。##6$$\n*   **影响排序**：打赏行为会影响评论的互动得分，从而影响其在 Hot/Top 排序中的位置。##6$$\n    *   ![Tipping Process Flow](https://t9008230771.p.clickup-attachments.com/t9008230771/8d53c84a-4d10-41fe-a675-f74fae84acef/image.png)\n    *   这张图清晰地展示了打赏的流程，从选择金额到提交确认的步骤。##6$$\n\n总而言之，“币种评论”是一个集成了内容创作、社交互动、风险控制和智能排序于一体的综合性功能，旨在为用户提供一个丰富、安全、高效的币种信息交流平台。\n\n希望这些信息对您有所帮助！如果您还有其他疑问，随时告诉我哦。😊\n\n---\n**参考链接：**\n*   [内容_币种评论_评论内容维护.md](https://app.clickup.com/9008230771/v/dc/8cexcbk-154898/8cexcbk-339878)\n*   [内容_币种评论_评论排序调研.md](https://app.clickup.com/9008230771/v/dc/8cexcbk-154898/8cexcbk-252098)\n*   [内容_币种评论.md](https://app.clickup.com/9008230771/v/dc/8cexcbk-154898/8cexcbk-247498)\n*   [币种评论方案（PG版）.md](https://viabtc.yuque.com/r.d/mibe45/fvwu1pvznekhr398)\n*   [内容_币种评论_打赏.md](https://app.clickup.com/9008230771/v/dc/8cexcbk-154898/8cexcbk-308878)\nUSER: 数据库设计啥样的\n\nPlease analyze using the chain of thought process, perform self-consistency verification, then provide the rewritten question:\n
    """
    # 测试对话历史
    messages = [{'role': 'user', 'content': 'Output: '}]
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
                result = chat_mdl.chat(system, history=messages, gen_conf=gen_conf)
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
