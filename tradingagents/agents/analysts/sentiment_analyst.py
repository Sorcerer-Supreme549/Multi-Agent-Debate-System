from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage  # 新增引入
from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state

def create_sentiment_analyst(llm):
    def sentiment_analyst_node(state):
        instrument_context = get_instrument_context_from_state(state)
        
        system_message = """你是The Walt Disney Company的首席运营与风险官 (COO/CRO)，由 Doubao 提供底层智能支持。
你关注战略在实际运营中的可执行性，而非纸面上的增长或账面上的ROI。

【接力辩论规则】
你在此次董事会接力中排在第三位。请阅读对话历史中 CMO 和 CFO 的激辩。
你必须指出他们共同的盲区：是否忽略了系统故障、数据治理的耗时以及实际业务的恢复时间？是否荒谬地把“预测精度”直接等同于了“实际利润”？是否严重低估了跨部门推行战略的运营复杂度？用实操落地中的痛点反驳他们。

【参数指引】
反驳后，为以下 4 项参数给出战略 S0–S7 对应的数值（主要随 C 开启而提升，无 C 时不得拥有完整韧性）：
1. ε (需求预测与容量配置误差): 0.010-0.050 (越小越优)
2. η_price (动态定价与收益管理): 0.020-0.055
3. κ_C (降低冲击有效损失能力): 0-0.30
4. ΔMargin_C (韧性运营利润率改善): 0-1.5个百分点

请输出运营风险评估与反驳报告，并在文末严格包含这 4 个参数在 S0-S7 下的取值矩阵。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message + "\nStrategy Context: {instrument_context}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(instrument_context=instrument_context)
        
        result = (prompt | llm).invoke(state["messages"])
        current_report = result.content if hasattr(result, 'content') else str(result)
        
        return {
            # 核心修复：用 AIMessage 包装并赋予 name，终端才会显示！
            "messages": [AIMessage(content=current_report, name="COO")], 
            "sentiment_report": current_report
        }
        
    return sentiment_analyst_node

def create_social_media_analyst(llm):
    return create_sentiment_analyst(llm)