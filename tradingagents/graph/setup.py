from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_portfolio_manager,
    create_sentiment_analyst,
    create_bull_researcher,
    create_bear_researcher
)
from tradingagents.agents.utils.agent_states import AgentState

class GraphSetup:
    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Any, 
        conditional_logic: Any,  
        analyst_concurrency_limit: int = 1,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm

    def setup_graph(self, selected_analysts=None):
        # 实例化 7 个壳节点
        market_node = create_market_analyst(self.quick_thinking_llm)         # 将作为 CMO_1
        fundamentals_node = create_fundamentals_analyst(self.quick_thinking_llm) # 将作为 CFO
        news_node = create_news_analyst(self.quick_thinking_llm)             # 将作为 CTO_1
        sentiment_node = create_sentiment_analyst(self.quick_thinking_llm)   # 将作为 COO
        bull_node = create_bull_researcher(self.quick_thinking_llm)          # 将作为 CMO_2
        bear_node = create_bear_researcher(self.quick_thinking_llm)          # 将作为 CTO_2
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm) # CEO

        workflow = StateGraph(AgentState)

        # ==========================================
        # 终极修复：严格使用框架原生的 Node Name 骗过 UI 拦截器！
        # 只要名字对得上，终端进度条就会完美点亮
        # ==========================================
        workflow.add_node("market_analyst", market_node)
        workflow.add_node("fundamentals_analyst", fundamentals_node)
        workflow.add_node("news_analyst", news_node)
        workflow.add_node("social_analyst", sentiment_node)
        workflow.add_node("Bull Researcher", bull_node)
        workflow.add_node("Bear Researcher", bear_node)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # ==========================================
        # 连线：构建极简无工具的 7 步单向接力拓扑
        # CMO_1 -> CFO -> CTO_1 -> COO -> CMO_2 -> CTO_2 -> CEO
        # ==========================================
        workflow.add_edge(START, "market_analyst")
        workflow.add_edge("market_analyst", "fundamentals_analyst")
        workflow.add_edge("fundamentals_analyst", "news_analyst")
        workflow.add_edge("news_analyst", "social_analyst")
        workflow.add_edge("social_analyst", "Bull Researcher")
        workflow.add_edge("Bull Researcher", "Bear Researcher")
        workflow.add_edge("Bear Researcher", "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", END)

        return workflow