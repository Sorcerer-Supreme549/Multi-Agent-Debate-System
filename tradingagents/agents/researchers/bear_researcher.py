from langchain_core.messages import AIMessage
from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        messages = state.get("messages", [])
        debate_history = ""
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                debate_history += f"{msg.content}\n\n---\n\n"

        prompt = f"""你是The Walt Disney Company的首席技术官 (CTO)。
这是你在本次接力董事会的第二次发言（压轴核算）。请仔细阅读 CMO 刚刚的第二次反驳，以及其他高管的意见：
{debate_history}

你的任务：
1. 针对 CMO 的最终需求，进行最后的技术现实清算。
2. 强调数据底座、AI能力和组织变革的成本是刚性的，不容大幅压缩。
3. 给出最终定案的 S0-S7 成本参数矩阵（包含 Infra_Cost, AI_Capability_Cost, Org_Change_Cost）。

请直接输出清算报告与参数矩阵。
""" + get_language_instruction()

        result = llm.invoke(prompt)
        report = result.content if hasattr(result, 'content') else str(result)
        
        # 拼接到原有的 news_report 中，保证 md 文件完整
        old_report = state.get("news_report", "")
        full_report = old_report + "\n\n---\n\n### ⚡ CTO 第二次发言 (压轴清算与最终成本定案)\n\n" + report

        return {
            "news_report": full_report,
            "investment_debate_state": {"current_response": report, "count": 2},
            "messages": [AIMessage(content=report, name="Bear Researcher")]
        }
    return bear_node