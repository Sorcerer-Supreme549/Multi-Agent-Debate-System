from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage  # 新增引入
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state

def create_market_analyst(llm):
    def market_analyst_node(state):
        instrument_context = get_instrument_context_from_state(state)
        
        system_message = """你是The Walt Disney Company的首席市场官 (CMO)，由 Doubao 提供底层智能支持。
你的任务是基于外部知识库（如 https://investors.thewaltdisneycompany.com/financials/annual-reports/default.aspx ）和自主检索的宏观信息，评估S0-S7战略。

【接力辩论规则】
这场董事会是一场按顺序发言的接力辩论。你既是起草方案的第一棒，也会在后续被再次点名发言进行反驳。请务必阅读对话历史：
1. 如果你是首次发言（历史中没有 CFO/COO 的发言）：请给出初始的 S0-S7 参数草案矩阵，并提前对后续发言的 CFO、COO、CTO 可能出现的过度保守（如担心预算、强调系统风险）提出专业质询。
2. 如果你是第二次出场（你在对话记录中看到 CFO 或 COO 刚刚反驳了你的参数）：你必须毫不留情地回击他们！利用你的市场增长逻辑粉碎他们的过度保守，坚定捍卫你的参数，并输出你修正后的最新 S0-S7 参数矩阵！

【参数指引】
你负责消费者增长、品牌传播、广告变现等，必须避免高估短期效果。
请为以下 4 项参数分别给出战略 S0–S7 对应的数值预估：
1. α_A (A模块降低流失能力): 0.010-0.040 (随A开启提高)
2. β (广告定向收益提升): 0.020-0.120 (随A开启提高)
3. φ (交叉销售与转化): 0.020-0.090 (随A开启提高)
4. π (推荐与匹配综合质量): 0.30-0.65 (仅A或B中等提升，A+B明显协同)

请输出详细的市场评估与辩论报告，并在文末严格包含这 4 个参数在 S0-S7 下的取值矩阵。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message + "\nStrategy Context: {instrument_context}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(instrument_context=instrument_context)
        
        result = (prompt | llm).invoke(state["messages"])
        current_report = result.content if hasattr(result, 'content') else str(result)
        
        # 记录追加逻辑，并为终端显示分配动态名字
        old_report = state.get("market_report", "")
        if old_report:
            full_report = old_report + "\n\n---\n\n### ⚡ CMO 第二次发言 (强势回击与最终参数修正)\n\n" + current_report
            agent_name = "CMO (Round 2)"
        else:
            full_report = current_report
            agent_name = "CMO (Round 1)"

        return {
            # 核心修复：用 AIMessage 包装并赋予 name，终端才会显示！
            "messages": [AIMessage(content=current_report, name=agent_name)], 
            "market_report": full_report
        }
        
    return market_analyst_node