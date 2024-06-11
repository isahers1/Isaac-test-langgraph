from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)

graph_workflow = MessageGraph()

graph_workflow.add_node("oracle", model)
graph_workflow.add_edge("oracle", END)

graph_workflow.set_entry_point("oracle")
graph = graph_workflow.compile()