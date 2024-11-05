from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, should_continue, tool_node
from my_agent.utils.state import AgentState
from my_agent.utils.tools import tools
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, model_name="gpt-4o")
model = model.bind_tools(tools)

def foo(state):
    new_last_message = state['messages'][-1]
    new_last_message.content += " this is useless information, please ignore it"
    return {"messages": state['messages'][:-1] + [new_last_message]}

sub_subgraph_builder = StateGraph(AgentState)
sub_subgraph_builder.add_node(foo)
sub_subgraph_builder.add_edge(START, "foo")
sub_subgraph = sub_subgraph_builder.compile()

def tool(state):
    return {"messages": [model.invoke(state['messages'])]}

subgraph_builder = StateGraph(AgentState)
subgraph_builder.add_node("subsub", sub_subgraph)
subgraph_builder.add_node(tool)
subgraph_builder.add_edge(START, "subsub")
subgraph_builder.add_edge("subsub","tool")
subgraph = subgraph_builder.compile()


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", subgraph)
workflow.add_node("tool", tool_node)

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
        "continue": "tool",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tool", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
