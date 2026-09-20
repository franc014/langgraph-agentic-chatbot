from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import os
import sqlite3


from langgraph.prebuilt import ToolNode, tools_condition
from tools import calculator, get_current_weather, get_stock_price, search_tool, rag_tool, make_rag_tool

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
MODEL_PROVIDER = "Groq"

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
    api_key=os.getenv('GROQ_API_KEY')
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def create_chatbot(retriever=None, document_name="uploaded PDF"):
    rag_search_tool = (
        make_rag_tool(retriever, document_name)
        if retriever
        else rag_tool
    )
    tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_search_tool]
    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state: ChatState):
    #take user query from state
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    graph = StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', ToolNode(tools))
    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')
    return graph.compile(checkpointer=checkpoint)


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpoint = SqliteSaver(conn)

chatbot = create_chatbot()


def get_all_threads():
    all_threads = set()
    #None: means bring all threads
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)




