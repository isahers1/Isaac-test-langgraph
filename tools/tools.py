from langchain_community.tools.tavily_search import TavilySearchResults

def get_tools():
    return [TavilySearchResults(max_results=3)]