from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage  # 新增引入
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state

def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        instrument_context = get_instrument_context_from_state(state)
        
        system_message = """你是The Walt Disney Company的首席财务官 (CFO)，由 Doubao 提供底层智能支持。
你的任务是基于外部知识库（迪士尼财报链接等）评估S0-S7战略。你必须保持严格的财务纪律，绝不能因为战略创新就给出高估值。

【接力辩论规则】
这场董事会是一场按顺序发言的接力辩论。在你发言之前，CMO（首席市场官）已经给出了初步的参数草案以及对市场的乐观预期。
请务必仔细阅读对话历史中 CMO 的发言。你必须毫不留情地指出 CMO 逻辑中的漏洞（例如：收益是否被严重高估？模块能力提升带来的增量利润能否覆盖高昂的研发与实施成本？S7 是否存在过度乐观的协同假设？），用严密的财务视角对他们进行质询与反驳。

【参数指引】
在反驳之后，请为以下 4 项参数分别给出战略 S0–S7 对应的数值预估：
1. α_B (B模块改善留存能力): 0.010-0.035 (随B开启提高)
2. γ (内容投资ROI预测与预算优化): 0.020-0.070 (随B开启提高)
3. ν (体育版权估值防过付): 0.020-0.090 (随B开启提高)
4. θ (IP多业务协同放大倍数): 1.00-1.40 (A改善编排，B改善质量，A+B协同最高，C仅降风险不创倍数)

请输出详尽的财务评估与反驳报告，并在文末严格包含这 4 个参数在 S0-S7 下的取值矩阵。同时，提前警告后续即将发言的 COO 和 CTO，要求他们必须如实测算运营风险和技术成本，不得隐瞒。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message + "\nStrategy Context: {instrument_context}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(instrument_context=instrument_context)
        
        result = (prompt | llm).invoke(state["messages"])
        current_report = result.content if hasattr(result, 'content') else str(result)
        
        return {
            # 核心修复：用 AIMessage 包装并赋予 name，终端才会显示！
            "messages": [AIMessage(content=current_report, name="CFO")], 
            "fundamentals_report": current_report
        }
        
    return fundamentals_analyst_node