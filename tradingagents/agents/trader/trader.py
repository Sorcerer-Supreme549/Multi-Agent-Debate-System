import functools
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)

def create_trader(llm):
    def trader_node(state, name):
        strategy_name = state.get("company_of_interest", "Disney 2026 Strategy")
        instrument_context = get_instrument_context_from_state(state)
        debate_history = state.get("investment_debate_state", {}).get("history", "No debate history.")

        system_message = (
            "You are the Strategy Integrator for the Walt Disney Company. "
            "Synthesize the debate and summarize the 15 strategy parameters boundaries. "
            "DO NOT use ANY stock market terminology. Focus entirely on maximizing 2026 profit. "
            + get_language_instruction()
        )

        user_message = (
            f"Strategy Context: {instrument_context}\n\n"
            f"Debate History:\n{debate_history}\n\n"
            f"Draft a unified 'Strategy Execution Proposal' for the CEO."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message),
        ])

        result = (prompt | llm).invoke({})
        trader_plan_content = result.content if hasattr(result, 'content') else str(result)
        final_output = f"### Disney 2026 Strategy Execution Proposal\n\n{trader_plan_content}"

        return {
            # 核心修复 1：提供 messages 让终端打印
            "messages": [AIMessage(content=final_output, name="Trader")],
            "trader_investment_plan": final_output,
            "sender": name,
        }

    # 核心修复 2：绝对不能修改 name="Trader"，否则 UI 进度条无法识别当前节点！
    return functools.partial(trader_node, name="Trader")