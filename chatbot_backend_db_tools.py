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
from tools import calculator, get_current_weather, get_stock_price, search_tool

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
    api_key=os.getenv('GROQ_API_KEY')
)

tools = [search_tool,calculator,get_stock_price, get_current_weather]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def chat_node(state: ChatState):
    #take user query from state
    messages = state['messages']
    # send to llm
    response = llm_with_tools.invoke(messages)
    # response store state
    return {'messages': [response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', ToolNode(tools))

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')


chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()
    #None: means bring all threads
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)






