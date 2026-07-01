from __future__ import annotations
import json
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state

def create_portfolio_manager(llm):
    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        
        # ==========================================
        # 核心操作：抽取底层最真实的 7 步流转对话记录
        # ==========================================
        messages = state.get("messages", [])
        real_time_history = ""
        step_counter = 1
        
        # 遍历所有底层消息，提取高管们的真实辩论内容
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                # 过滤掉系统短消息，只保留高管的论战长文
                if len(str(msg.content)) > 50:
                    real_time_history += f"#### 【第 {step_counter} 棒辩论现场】\n{msg.content}\n\n---\n\n"
                    step_counter += 1

        # ==========================================
        # CEO 裁决逻辑
        # ==========================================
        system_message = """你是The Walt Disney Company的首席执行官兼战略投资委员会主席 Bob Iger。
你的任务是整合各方观点，终审输出 2026年 S0-S7 战略的最终参数矩阵。"""
        
        prompt_text = f"""
【会议完整录音】：
{real_time_history}

【战略背景】：{instrument_context}

作为 CEO，请执行以下流程：
第一部分：董事会观点整合与决策过程
请根据会议记录，总结 CMO、CFO、COO、CTO 关于收益与成本的激烈分歧。明确指出你在哪些参数上支持了 CMO，在哪些参数上采纳了 CFO/CTO 的成本把控，并阐述你进行裁决的商业逻辑。

第二部分：最终参数矩阵
在分析之后，输出一个包含完整 S0-S7 参数的标准 JSON 矩阵（使用 ```json 块包围）。
必须包含 "S0" 到 "S7" 的 keys，且每个均包含 15 个参数。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", prompt_text)
        ])
        
        result = (prompt | llm).invoke({})
        ceo_verdict = result.content if hasattr(result, 'content') else str(result)

        # ==========================================
        # 终极修复：把会议录音带和 CEO 裁决物理缝合
        # 这样无论框架怎么渲染，这份报告永远是按时间顺序展现的！
        # ==========================================
        full_final_report = (
            "## 🎬 迪士尼 2026 战略董事会：7 步接力圆桌会议全纪要\n"
            "*(以下内容严格按照 CMO -> CFO -> CTO -> COO -> CMO -> CTO 的真实发言顺序记录)*\n\n"
            f"{real_time_history}\n\n"
            "## 👑 CEO (Bob Iger) 最终整合与裁决\n\n"
            f"{ceo_verdict}"
        )

        new_risk_debate_state = state.get("risk_debate_state", {})
        new_risk_debate_state["judge_decision"] = full_final_report

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": full_final_report,
            "messages": [AIMessage(content=full_final_report, name="Portfolio Manager")]
        }

    return portfolio_manager_node