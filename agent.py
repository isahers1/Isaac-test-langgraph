from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage
from pydantic.v1 import BaseModel, Field


def update_user_info(old_info, new_info):
    if 'name' not in new_info or new_info['age'] == -1:
        return old_info
    return new_info

class UserInformation(TypedDict):
    age: int
    name: str

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: Annotated[UserInformation, update_user_info]

class PersonalInfo(BaseModel):
    """Personal information of the user"""

    age: int = Field(description="The age of the user, default to -1 if you cannot find their age")
    name: str = Field(description="The name of the user, default to John Doe if you can't find their name")


base_model = ChatOpenAI(temperature=0)
model = ChatOpenAI(temperature=0)
model = model.bind_tools([PersonalInfo],tool_choice="PersonalInfo")

def run_llm(state: AgentState) -> AgentState:
    response = model.invoke(state['messages'][-1].content)
    return {"messages":[response]}

def run_chat_llm(state):
    try:
        extra_message = [SystemMessage(content=f"Recall that the user is {state['user_info']['age']} years old and their name is {state['user_info']['name']}. Make sure to use this information if it is relevant to the last user query.")]
    except:
        extra_message = []

    ai_response = base_model.invoke([message for message in state['messages']  if len(message.content) > 0]+extra_message)
    return {"messages":[ai_response]}

def tool_node(state: AgentState) -> AgentState:
    last_message = state['messages'][-1]
    tool_call = last_message.tool_calls[0]
    return {"messages":state["messages"],"user_info":UserInformation(tool_call['args'])}

graph_workflow = StateGraph(AgentState)

graph_workflow.add_node("llm", run_llm)
graph_workflow.add_node("get_user_info", tool_node)
graph_workflow.add_node("respond_to_user", run_chat_llm)
graph_workflow.add_edge("llm", "get_user_info")
graph_workflow.add_edge("get_user_info", "respond_to_user")
graph_workflow.add_edge("respond_to_user", END)
graph_workflow.set_entry_point("llm")
graph = graph_workflow.compile()