from app.generative import manager as AiManager
from app.utils.HttpResponseUtils import response_success, response_error
from app.services.ChatbotRouterService import ChatbotRouter
from app.services.DoctorAgentService import DoctorAgent
from app.services.QNAAgentService import QNAAgent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict
from config.mcp import mcpconfig
import traceback
# Define state
class AgentState(TypedDict):
    input: dict
    decision: str
    response: str


class ChatbotController:
    """
    Controller untuk chatbot dengan router yang menentukan routing ke doctor agent atau qna agent.
    """
    
    def __init__(self):
        """
        Initialize ChatbotController.
        
        Args:
            mcp_server_key: Key untuk MCP server (optional, untuk doctor agent)
        """
        self.llm = AiManager.gemini_mini()  # gemini flash lite 2.5
        self.mcp_config = mcpconfig
        # Initialize router
        self.router_agent = ChatbotRouter(llm=self.llm)
        
        # Initialize sub-agents
        self.doctor_agent = DoctorAgent(llm=self.llm, mcp_config=self.mcp_config)
        
        # Initialize QNA agent dengan lazy loading tools
        # Tools akan di-load saat pertama kali digunakan (setelah open() dipanggil)
        mcp_tools = self.mcp_config.get_tools_for_bind(["faq", "hospital"]) if self.mcp_config else []
        self.qna_agent = QNAAgent(
            llm=self.llm,
            tools_mcp=mcp_tools
        )

        self.build_graph(checkpoint=InMemorySaver())

    
    async def doctor_node(self, state: AgentState) -> Dict:
        """
        Node untuk doctor agent dengan tool execution.
        """
        try:
            result = await self.doctor_agent(state)
            return result
        except Exception as e:
            traceback.print_exc()
            return response_error(str(e))
    
    async def qna_node(self, state: AgentState) -> Dict:
        """
        Node untuk QNA agent dengan tool execution.
        """
        try:
            result = await self.qna_agent(state)
            return result
        except Exception as e:
            traceback.print_exc()
            return response_error(str(e))

    
    def build_graph(self, checkpoint=None):
        """
        Build LangGraph workflow dengan router, doctor agent, dan qna agent.
        
        Args:
            checkpoint: Optional checkpoint untuk graph persistence
        
        Returns:
            Compiled graph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_agent)
        workflow.add_node("doctor", self.doctor_node)
        workflow.add_node("qna", self.qna_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",  # starting point
            lambda state: state.get('decision', 'end'),  # decision state
            {
                # Mapping decision to target nodes
                "doctor": "doctor",
                "qna": "qna",
                "end": END
            }
        )
        
        # Add edges to END
        workflow.add_edge("doctor", END)
        workflow.add_edge("qna", END)
        
        # Compile graph
        graph = workflow.compile(checkpointer=checkpoint)
        self.graph = graph
        return graph
    
    async def start_chatting(self, input_data: dict):
        """
        Start chatting dengan input dari user.
        
        Args:
            input_data: Dictionary dengan "text" key berisi user query
        
        Returns:
            Response dari agent yang dipilih
        """
        
        try:
            initial_state = AgentState(
                input=input_data,
                decision="",
                response=""
            )
            config = {'configurable': {'thread_id': '123'}}
            result = await self.graph.ainvoke(initial_state, config=config)
            print(f"Chatbot Result: {result}")
            return response_success(result['response'])
        except Exception as e:
            traceback.print_exc()
            return response_error(str(e))


# Create instance
chatbotController = ChatbotController()
