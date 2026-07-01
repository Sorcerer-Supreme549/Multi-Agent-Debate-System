from langchain_core.messages import AIMessage
from tradingagents.agents.utils.agent_utils import get_language_instruction

def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        messages = state.get("messages", [])
        debate_history = ""
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                debate_history += f"{msg.content}\n\n---\n\n"

        prompt = f"""你是The Walt Disney Company的首席市场官 (CMO)。
这是你在本次接力董事会的第二次发言。请仔细阅读之前你自己的初始草案，以及随后 CFO、CTO、COO 对你的财务、技术和运营质询：
{debate_history}

你的任务：
1. 强烈反驳 CFO、COO、CTO 关于成本和风险的过度保守观点。
2. 坚持你的市场增长逻辑，说明为什么速度和跨业务协同是不可妥协的。
3. 给出你最终修正后的 S0-S7 参数草案矩阵（包含 α_A, β, φ, π）。

请直接输出反驳报告与参数矩阵。
""" + get_language_instruction()

        result = llm.invoke(prompt)
        report = result.content if hasattr(result, 'content') else str(result)
        
        # 拼接到原有的 market_report 中，保证 md 文件完整无缺
        old_report = state.get("market_report", "")
        full_report = old_report + "\n\n---\n\n### ⚡ CMO 第二次发言 (强势回击与参数修正)\n\n" + report

        return {
            "market_report": full_report,
            "investment_debate_state": {"current_response": report, "count": 1},
            "messages": [AIMessage(content=report, name="Bull Researcher")]
        }
    return bull_node