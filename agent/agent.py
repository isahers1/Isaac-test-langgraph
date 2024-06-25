from langgraph.graph import END, StateGraph
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from tools.tools import get_tools

tools = get_tools()
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o",)

model = llm.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke([SystemMessage(
        content="You are a helpful assistant! Please use the tool at your disposal if the user asks for weather information."
    ),]+messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("search", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "search",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("search", "agent")
graph = workflow.compile()


'''

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


anthropic_model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
openai_model = ChatOpenAI(temperature=0)
model = ChatOpenAI(temperature=0)
model = model.bind_tools([PersonalInfo],tool_choice="PersonalInfo")

chat_models = {
    "anthropic": anthropic_model,
    "openai": openai_model,
}

def run_llm(state: AgentState) -> AgentState:
    response = model.invoke(state['messages'][-1].content)
    return {"messages":[response]}

def run_chat_llm(state, config):
    try:
        extra_message = [SystemMessage(content=f"Recall that the user is {state['user_info']['age']} years old and their name is {state['user_info']['name']}. Make sure to use this information if it is relevant to the last user query.")]
    except:
        extra_message = []

    m = chat_models[config["configurable"].get("model", "anthropic")]
    ai_response = m.invoke([message for message in state['messages']  if len(message.content) > 0]+extra_message)
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

'''