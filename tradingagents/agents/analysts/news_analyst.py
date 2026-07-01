from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage  # 新增引入
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state

def create_news_analyst(llm):
    def news_analyst_node(state):
        instrument_context = get_instrument_context_from_state(state)
        
        system_message = """你是The Walt Disney Company的首席技术官 (CTO)，由 Doubao 提供底层智能支持。
你负责数据平台、AI模型与系统集成的投入评估。你不负责估算市场收益，只负责核算冰冷的成本账单。

【接力辩论规则】
你在此次圆桌会议中会出场核算技术成本。仔细审阅前面所有人的论战。
不论前面的 CMO 和 CFO 把协同效应吹得多高，你必须用技术现实击碎他们的幻想：毫不留情地指出他们是否低估了数据清洗、系统连接和模型维护的长期隐性成本？是否忽略了培训、治理和流程重构的巨大阻力？

【参数指引】
反驳后，为以下 3 项成本给出战略 S0–S7 对应的数值（S0原则上为0）：
1. Infra_Cost (数据底座新增投入): $0-250m (由A和C驱动)
2. AI_Capability_Cost (AI模型/数字孪生投入): $0-200m (由B和C驱动)
3. Org_Change_Cost (组织变革流程重构投入): $0-150m (由A和C驱动)
注意：共享基础设施可降低重复投资，但集成复杂度会推高成本，不得简单相加。

请输出详尽的技术成本清算与反驳报告，并在文末严格包含这 3 个成本参数在 S0-S7 下的取值矩阵，以供最后出场的 CEO 进行全盘裁决。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message + "\nStrategy Context: {instrument_context}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(instrument_context=instrument_context)
        
        result = (prompt | llm).invoke(state["messages"])
        current_report = result.content if hasattr(result, 'content') else str(result)
        
        old_report = state.get("news_report", "")
        if old_report:
            full_report = old_report + "\n\n---\n\n### ⚡ CTO 第二次发言 (压轴清算与最终成本定案)\n\n" + current_report
            agent_name = "CTO (Round 2)"
        else:
            full_report = current_report
            agent_name = "CTO (Round 1)"

        return {
            # 核心修复：用 AIMessage 包装并赋予 name，终端才会显示！
            "messages": [AIMessage(content=current_report, name=agent_name)], 
            "news_report": full_report
        }
        
    return news_analyst_node