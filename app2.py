import os
import threading
import time
import requests
import json
import sys
import logging
from langgraph.graph import END
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState, A2AClient,Task
from python_a2a.mcp import FastMCP, text_response
from langchain.tools import Tool
from langgraph.graph import StateGraph

def create_task(message):
    task = Task(message=message)
    task.message_id = str(uuid.uuid4())       # Unique identifier for the task
    task.conversation_id = str(uuid.uuid4())  # Unique identifier for the conversation
    return task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import TypedDict, List

class GraphState(TypedDict):
    messages: List[str]
    cycle_count: int
    should_continue: bool

@dataclass
class ServerInfo:
    name: str
    port: int
    status: str
    thread: Optional[threading.Thread] = None

class GroqAPIClient:
    """Groq API client for LLM calls"""
    def __init__(self, api_key: str):
        self.groq_api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def call_groq_api(self, messages: list, max_retries: int = 3) -> str:
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        payload = {"model": "deepseek-r1-distill-llama-70b", "messages": messages, "max_tokens": 1500, "temperature": 0.3}
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                logger.error(f"Groq API call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return "Error: Failed to get response from Groq API"

class MCPToolsManager:
    """Manager for MCP Tools to LangChain conversion"""
    def __init__(self, groq_client: GroqAPIClient):
        self.groq_client = groq_client
    
    def create_mcp_calculator_server(self, port: int = 5001):
        mcp_server = FastMCP(name="Calculator Tools", description="Mathematical calculation tools")
        @mcp_server.tool(name="calculator", description="Calculate a mathematical expression safely")
        def calculator(expression: str):
            try:
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return text_response("Error: Invalid characters in expression")
                result = eval(expression)
                return text_response(f"Result: {expression} = {result}")
            except Exception as e:
                return text_response(f"Error: {str(e)}")
        return mcp_server
    
    def create_mcp_text_analyzer_server(self, port: int = 5002):
        mcp_server = FastMCP(name="Text Analyzer", description="Text analysis and processing tools")
        @mcp_server.tool(name="text_analyzer", description="Analyze text and provide insights")
        def text_analyzer(text: str):
            try:
                words = len(text.split())
                chars = len(text)
                sentences = text.count('.') + text.count('!') + text.count('?')
                analysis = {
                    "word_count": words,
                    "character_count": chars,
                    "sentence_count": sentences,
                    "average_word_length": round(chars / words, 2) if words > 0 else 0
                }
                return text_response(f"Text Analysis: {json.dumps(analysis, indent=2)}")
            except Exception as e:
                return text_response(f"Error: {str(e)}")
        return mcp_server
    
    def mcp_to_langchain_tool(self, server_url: str, tool_name: str):
        def tool_function(input_text: str) -> str:
            try:
                if tool_name == "calculator":
                    payload = {"expression": input_text}
                elif tool_name == "text_analyzer":
                    payload = {"text": input_text}
                else:
                    payload = {"input": input_text}
                
                response = requests.post(f"{server_url}/tools/{tool_name}", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                
                if "result" in result:
                    if isinstance(result["result"], dict) and "content" in result["result"]:
                        return result["result"]["content"]
                    else:
                        return str(result["result"])
                else:
                    return str(result)
            except Exception as e:
                return f"Error calling MCP tool: {str(e)}"
        return Tool(name=tool_name, description=f"MCP tool: {tool_name}", func=tool_function)

class LangChainToMCPConverter:
    """Convert LangChain tools to MCP servers"""
    def __init__(self, groq_client: GroqAPIClient):
        self.groq_client = groq_client
    
    def create_langchain_tools(self):
        def weather_tool(location: str) -> str:
            return f"Weather in {location}: Sunny, 75°F with light breeze"
        
        def translator_tool(text_and_lang: str) -> str:
            try:
                parts = text_and_lang.split(" to ")
                if len(parts) != 2:
                    return "Error: Format should be 'text to language'"
                text, target_lang = parts
                messages = [{"role": "user", "content": f"Translate '{text}' to {target_lang}"}]
                return self.groq_client.call_groq_api(messages)
            except Exception as e:
                return f"Translation error: {str(e)}"
        
        return [
            Tool(name="weather", description="Get weather for a location", func=weather_tool),
            Tool(name="translator", description="Translate text to another language", func=translator_tool)
        ]
    
    def langchain_to_mcp_server(self, langchain_tools: List[Tool]):
        mcp_server = FastMCP(name="LangChain Tools", description="Converted LangChain tools")
        
        for tool in langchain_tools:
            def create_mcp_tool(lc_tool):
                @mcp_server.tool(name=lc_tool.name, description=lc_tool.description)
                def mcp_tool_func(input_data: str):
                    try:
                        result = lc_tool.func(input_data)
                        return text_response(str(result))
                    except Exception as e:
                        return text_response(f"Error: {str(e)}")
                return mcp_tool_func
            create_mcp_tool(tool)
        
        return mcp_server

@agent(name="Directory Agent", description="Directory for agent discovery", version="1.0.0")
class DirectoryAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.registered_agents = {}
    
    @skill(name="register_agent", description="Register an agent")
    def register_agent(self, name: str, description: str, url: str):
        self.registered_agents[url] = {"name": name, "description": description, "url": url}
        return "Agent registered successfully"
    
    @skill(name="list_agents", description="List registered agents")
    def list_agents(self):
        return json.dumps(list(self.registered_agents.values()), indent=2)
    
    @skill(name="get_capabilities", description="Get the list of capabilities")
    def get_capabilities(self):
        return json.dumps({"skills": ["register_agent", "list_agents", "get_capabilities"]}, indent=2)
    
    def handle_task(self, task):
        try:
            message = task.message
            if isinstance(message, str):
                message = json.loads(message)
            skill_name = message.get("skill")
            if skill_name == "register_agent":
                name = message.get("name")
                description = message.get("description")
                url = message.get("url")
                response = self.register_agent(name, description, url)
            elif skill_name == "list_agents":
                response = self.list_agents()
            elif skill_name == "get_capabilities":
                response = self.get_capabilities()
            else:
                response = "Unknown skill"
            
            task.artifacts = [{"parts": [{"type": "text", "text": response}]}]
            task.status = TaskStatus(state=TaskState.COMPLETED)
        except Exception as e:
            task.status = TaskStatus(state=TaskState.FAILED, message={
                "role": "agent", 
                "content": {"type": "text", "text": f"Error: {str(e)}"}
            })
        return task

class LangChainToA2AConverter:
    """Convert LangChain components to A2A servers"""
    def __init__(self, groq_client: GroqAPIClient):
        self.groq_client = groq_client
    
    def create_groq_llm_agent(self):
        @agent(name="Groq LLM Agent", description="LLM agent powered by Groq API", version="1.0.0")
        class GroqLLMAgent(A2AServer):
            def __init__(self, groq_client):
                super().__init__()
                self.groq_client = groq_client
            
            @skill(name="chat", description="Chat using Groq LLM", tags=["chat", "llm"])
            def chat(self, query: str):
                messages = [{"role": "user", "content": query}]
                return self.groq_client.call_groq_api(messages)
            
            @skill(name="get_capabilities", description="Get the list of capabilities")
            def get_capabilities(self):
                return json.dumps({"skills": ["chat", "get_capabilities"]}, indent=2)
            
            def handle_task(self, task):
                try:
                    message = task.message
                    if isinstance(message, str):
                        message = json.loads(message)
                    skill_name = message.get("skill")
                    if skill_name == "chat":
                        query = message.get("query")
                        response = self.chat(query)
                    elif skill_name == "get_capabilities":
                        response = self.get_capabilities()
                    else:
                        response = "Unknown skill"
                    
                    task.artifacts = [{"parts": [{"type": "text", "text": response}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)
                except Exception as e:
                    task.status = TaskStatus(state=TaskState.FAILED, message={
                        "role": "agent", 
                        "content": {"type": "text", "text": f"Error: {str(e)}"}
                    })
                return task
        
        return GroqLLMAgent(self.groq_client)
    
    def create_travel_guide_agent(self):
        @agent(name="Travel Guide Agent", description="Specialized travel guide with Groq LLM", version="1.0.0")
        class TravelGuideAgent(A2AServer):
            def __init__(self, groq_client):
                super().__init__()
                self.groq_client = groq_client
            
            @skill(name="travel_advice", description="Get travel advice and recommendations", tags=["travel", "guide", "recommendations"])
            def get_travel_advice(self, query: str):
                messages = [
                    {"role": "system", "content": "You are a helpful travel guide with extensive knowledge of destinations worldwide."},
                    {"role": "user", "content": query}
                ]
                return self.groq_client.call_groq_api(messages)
            
            @skill(name="get_capabilities", description="Get the list of capabilities")
            def get_capabilities(self):
                return json.dumps({"skills": ["travel_advice", "get_capabilities"]}, indent=2)
            
            def handle_task(self, task):
                try:
                    message = task.message
                    if isinstance(message, str):
                        message = json.loads(message)
                    skill_name = message.get("skill")
                    if skill_name == "travel_advice":
                        query = message.get("query")
                        response = self.get_travel_advice(query)
                    elif skill_name == "get_capabilities":
                        response = self.get_capabilities()
                    else:
                        response = "Unknown skill"
                    
                    task.artifacts = [{"parts": [{"type": "text", "text": response}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)
                except Exception as e:
                    task.status = TaskStatus(state=TaskState.FAILED, message={
                        "role": "agent", 
                        "content": {"type": "text", "text": f"Error: {str(e)}"}
                    })
                return task
        
        return TravelGuideAgent(self.groq_client)

class A2AToLangChainConverter:
    """Convert A2A agents to LangChain tools"""
    def __init__(self, groq_client: GroqAPIClient):
        self.groq_client = groq_client
    
    def a2a_to_langchain_tool(self, agent_url: str, agent_name: str):
        def tool_function(query: str) -> str:
            try:
                client = A2AClient(agent_url)
                response = client.ask({"skill": "chat", "query": query})
                return str(response)
            except Exception as e:
                return f"Error calling A2A agent: {str(e)}"
        
        return Tool(
            name=agent_name.lower().replace(" ", "_"), 
            description=f"A2A agent: {agent_name}", 
            func=tool_function
        )

class LangGraphIntegration:
    """LangGraph integration with Groq API"""
    def __init__(self, groq_client: GroqAPIClient):
        self.groq_client = groq_client
    
    def create_simple_graph(self, tools: List[Tool]):
        def llm_node(state):
            messages = state.get("messages", [])
            cycle_count = state.get("cycle_count", 0)
            
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            conversation_history = "\n".join(messages)
            prompt = f"""You are an assistant with access to the following tools:

{tool_descriptions}

Instructions:
- When you decide to use a tool, respond with exactly: USE_TOOL: <tool_name> <input>
- When you have the final answer, respond directly with the answer.
- If the conversation history includes a 'Tool result:', use that result to provide the final answer.

Important:
- When using a tool, your response must be solely the USE_TOOL command, with no other text, explanations, or tags like <think>.
- Do not include any additional text or tags in your response.

Conversation history:
{conversation_history}

Now, provide your response:"""
            
            response = self.groq_client.call_groq_api([{"role": "user", "content": prompt}])
            return {"messages": messages + [response], "cycle_count": cycle_count + 1}
        
        def tool_node(state):
            messages = state.get("messages", [])
            if not messages:
                return {"messages": [], "should_continue": False}
            
            last_message = messages[-1]
            # Look for a line that starts with "USE_TOOL:"
            for line in last_message.splitlines():
                line = line.strip()
                if line.startswith("USE_TOOL:"):
                    try:
                        _, tool_info = line.split(":", 1)
                        tool_name, tool_input = tool_info.strip().split(" ", 1)
                        for tool in tools:
                            if tool.name.lower() == tool_name.lower():
                                result = tool.func(tool_input)
                                return {"messages": messages + [f"Tool result: {result}"], "should_continue": True}
                        return {"messages": messages + ["Error: Tool not found"], "should_continue": False}
                    except Exception as e:
                        return {"messages": messages + [f"Error parsing tool request: {str(e)}"], "should_continue": False}
            
            # If no USE_TOOL command found, assume it's the final answer
            return {"messages": messages, "should_continue": False}
        
        def route_tools(state):
            if state.get("should_continue", False) and state.get("cycle_count", 0) < 5:
                return "continue"
            return "stop"
        
        graph = StateGraph(GraphState)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node)
        graph.add_edge("llm", "tools")
        graph.add_conditional_edges(
            "tools",
            route_tools,
            {"continue": "llm", "stop": END}
        )
        graph.set_entry_point("llm")
        
        return graph.compile()

class ServerManager:
    """Manage all servers and their lifecycle"""
    def __init__(self):
        self.servers = {}
    
    def start_mcp_server(self, name: str, server_instance, port: int):
        def run_server():
            try:
                server_instance.run(host="0.0.0.0", port=port)
            except Exception as e:
                logger.error(f"Error running MCP server {name}: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.servers[name] = ServerInfo(name, port, "running", thread)
        time.sleep(2)  # Give server time to start
        return f"http://localhost:{port}"
    
    import json

    def start_a2a_server(self, name: str, agent_instance, port: int):
        url = f"http://localhost:{port}"
        # Register with DirectoryAgent only if not the Directory Agent itself
        if name != "Directory Agent":
            try:
                directory_client = A2AClient("http://localhost:5000")
                registration_task = create_task({
                    "skill": "register_agent",
                    "name": name,
                    "description": agent_instance.description,
                    "url": url
                })
                response = directory_client.ask(registration_task)
                print(f"Registered {name} with DirectoryAgent: {response}")
            except Exception as e:
                print(f"Error registering {name} with DirectoryAgent: {e}")
        
        def run_server():
            try:
                agent_instance.run(host="0.0.0.0", port=port)
            except Exception as e:
                logger.error(f"Error running A2A server {name}: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.servers[name] = ServerInfo(name, port, "running", thread)
        time.sleep(2)  # Give server time to start
        return url
    
    def is_running(self, name: str) -> bool:
        if name not in self.servers:
            return False
        return self.servers[name].thread and self.servers[name].thread.is_alive()
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        status = {}
        for name, info in self.servers.items():
            is_alive = info.thread and info.thread.is_alive()
            status[name] = {
                "port": info.port,
                "status": "running" if is_alive else "stopped"
            }
        return status
    
    def stop_server(self, name: str):
        if name in self.servers:
            del self.servers[name]
            print(f"Server {name} stopped")

def main():
    print("Welcome to MCP-A2A-LangChain Integration Tool")
    
    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable is required")
        print("Please set it using: export GROQ_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Initialize components
    groq_client = GroqAPIClient(groq_api_key)
    server_manager = ServerManager()
    
    # Start DirectoryAgent
    directory_agent = DirectoryAgent()
    directory_url = server_manager.start_a2a_server("Directory Agent", directory_agent, 5000)
    print(f"Directory Agent started at {directory_url}")
    
    # Context to store shared objects
    context = {
        "groq_client": groq_client,
        "server_manager": server_manager,
        "langgraph": None,
        "a2a_langchain_tools": [],
    }
    
    try:
        while True:
            print("\nMain Menu:")
            print("0. View Server Status")
            print("1. Start MCP Servers")
            print("2. Test MCP to LangChain Conversion")
            print("3. Start LangChain Tools as MCP Server")
            print("4. Test Converted LangChain Tools")
            print("5. Start A2A Agents")
            print("6. Test A2A Agents")
            print("7. Create LangChain Tools from A2A Agents")
            print("8. Test LangChain Tools from A2A")
            print("9. Create LangGraph with Tools")
            print("10. Test LangGraph")
            print("11. Discover Agents and Capabilities")
            print("12. Exit")
            
            choice = input("Enter your choice: ").strip()
            
            if choice == "0":
                status = server_manager.get_server_status()
                print("\nServer Status:")
                if not status:
                    print("No servers running")
                else:
                    for name, info in status.items():
                        print(f"{name}: {info['status']} on port {info['port']}")
            
            elif choice == "1":
                while True:
                    print("\nStart MCP Servers:")
                    print("1. Start Calculator MCP Server")
                    print("2. Start Text Analyzer MCP Server")
                    print("3. Back to Main Menu")
                    sub_choice = input("Enter your choice: ").strip()
                    
                    if sub_choice == "1":
                        if server_manager.is_running("Calculator MCP"):
                            print("Calculator MCP Server is already running")
                        else:
                            try:
                                manager = MCPToolsManager(groq_client)
                                server = manager.create_mcp_calculator_server()
                                url = server_manager.start_mcp_server("Calculator MCP", server, 5001)
                                print(f"Calculator MCP Server started at {url}")
                            except Exception as e:
                                print(f"Error starting Calculator MCP Server: {e}")
                    
                    elif sub_choice == "2":
                        if server_manager.is_running("Text Analyzer MCP"):
                            print("Text Analyzer MCP Server is already running")
                        else:
                            try:
                                manager = MCPToolsManager(groq_client)
                                server = manager.create_mcp_text_analyzer_server()
                                url = server_manager.start_mcp_server("Text Analyzer MCP", server, 5002)
                                print(f"Text Analyzer MCP Server started at {url}")
                            except Exception as e:
                                print(f"Error starting Text Analyzer MCP Server: {e}")
                    
                    elif sub_choice == "3":
                        break
                    else:
                        print("Invalid choice")
            
            elif choice == "2":
                print("\nTest MCP to LangChain Conversion")
                
                # Test Calculator
                if not server_manager.is_running("Calculator MCP"):
                    print("Calculator MCP Server is not running. Please start it first.")
                else:
                    test_expression = input("Enter a mathematical expression (e.g., '2 + 3 * 4'): ").strip()
                    try:
                        manager = MCPToolsManager(groq_client)
                        tool = manager.mcp_to_langchain_tool("http://localhost:5001", "calculator")
                        result = tool.run(test_expression)
                        print(f"Calculator Result: {result}")
                    except Exception as e:
                        print(f"Calculator Error: {e}")
                
                # Test Text Analyzer
                if not server_manager.is_running("Text Analyzer MCP"):
                    print("Text Analyzer MCP Server is not running. Please start it first.")
                else:
                    test_text = input("Enter text to analyze (e.g., 'Hello world! This is a test.'): ").strip()
                    try:
                        manager = MCPToolsManager(groq_client)
                        tool = manager.mcp_to_langchain_tool("http://localhost:5002", "text_analyzer")
                        result = tool.run(test_text)
                        print(f"Text Analysis Result: {result}")
                    except Exception as e:
                        print(f"Text Analysis Error: {e}")
            
            elif choice == "3":
                if server_manager.is_running("LangChain MCP"):
                    print("LangChain MCP Server is already running")
                else:
                    try:
                        converter = LangChainToMCPConverter(groq_client)
                        tools = converter.create_langchain_tools()
                        server = converter.langchain_to_mcp_server(tools)
                        url = server_manager.start_mcp_server("LangChain MCP", server, 5003)
                        print(f"LangChain MCP Server started at {url}")
                        print("Available tools: weather, translator")
                    except Exception as e:
                        print(f"Error starting LangChain MCP server: {e}")
            
            elif choice == "4":
                print("\nTest Converted LangChain Tools")
                if not server_manager.is_running("LangChain MCP"):
                    print("LangChain MCP Server is not running. Please start it first.")
                else:
                    # Test Weather
                    location = input("Enter location for weather (e.g., 'New York'): ").strip()
                    try:
                        response = requests.post("http://localhost:5003/tools/weather", 
                                               json={"input_data": location}, timeout=10)
                        if response.status_code == 200:
                            result = response.json()
                            print(f"Weather Result: {result}")
                        else:
                            print(f"Weather Error: HTTP {response.status_code}")
                    except Exception as e:
                        print(f"Weather Error: {e}")
                    
                    # Test Translator
                    translation = input("Enter text to translate (e.g., 'Hello to Spanish'): ").strip()
                    try:
                        response = requests.post("http://localhost:5003/tools/translator", 
                                               json={"input_data": translation}, timeout=15)
                        if response.status_code == 200:
                            result = response.json()
                            print(f"Translation Result: {result}")
                        else:
                            print(f"Translation Error: HTTP {response.status_code}")
                    except Exception as e:
                        print(f"Translation Error: {e}")
            
            elif choice == "5":
                while True:
                    print("\nStart A2A Agents:")
                    print("1. Start Groq LLM Agent")
                    print("2. Start Travel Guide Agent")
                    print("3. Back to Main Menu")
                    sub_choice = input("Enter your choice: ").strip()
                    
                    if sub_choice == "1":
                        if server_manager.is_running("Groq LLM Agent"):
                            print("Groq LLM Agent is already running")
                        else:
                            try:
                                converter = LangChainToA2AConverter(groq_client)
                                agent = converter.create_groq_llm_agent()
                                url = server_manager.start_a2a_server("Groq LLM Agent", agent, 5004)
                                print(f"Groq LLM Agent started at {url}")
                            except Exception as e:
                                print(f"Error starting Groq LLM Agent: {e}")
                    
                    elif sub_choice == "2":
                        if server_manager.is_running("Travel Guide Agent"):
                            print("Travel Guide Agent is already running")
                        else:
                            try:
                                converter = LangChainToA2AConverter(groq_client)
                                agent = converter.create_travel_guide_agent()
                                url = server_manager.start_a2a_server("Travel Guide Agent", agent, 5005)
                                print(f"Travel Guide Agent started at {url}")
                            except Exception as e:
                                print(f"Error starting Travel Guide Agent: {e}")
                    
                    elif sub_choice == "3":
                        break
                    else:
                        print("Invalid choice")
            
            elif choice == "6":
                print("\nTest A2A Agents")
                
                # Test Groq LLM Agent
                if not server_manager.is_running("Groq LLM Agent"):
                    print("Groq LLM Agent is not running. Please start it first.")
                else:
                    llm_query = input("Enter query for Groq LLM Agent (e.g., 'What is artificial intelligence?'): ").strip()
                    try:
                        client = A2AClient("http://localhost:5004")
                        chat_task = create_task({"skill": "chat", "query": llm_query})
                        response = client.ask(chat_task)
                        print(f"LLM Agent Response: {response}")
                    except Exception as e:
                        print(f"LLM Agent Error: {e}")
                
                # Test Travel Guide Agent
                if not server_manager.is_running("Travel Guide Agent"):
                    print("Travel Guide Agent is not running. Please start it first.")
                else:
                    travel_query = input("Enter query for Travel Guide Agent (e.g., 'Best places to visit in Japan'): ").strip()
                    try:
                        client = A2AClient("http://localhost:5005")
                        advice_task = create_task({"skill": "travel_advice", "query": travel_query})
                        response = client.ask(advice_task)
                        print(f"Travel Guide Response: {response}")
                    except Exception as e:
                        print(f"Travel Guide Error: {e}")
            
            elif choice == "7":
                try:
                    converter = A2AToLangChainConverter(groq_client)
                    tools = []
                    
                    if server_manager.is_running("Groq LLM Agent"):
                        llm_tool = converter.a2a_to_langchain_tool("http://localhost:5004", "Groq LLM Agent")
                        tools.append(llm_tool)
                    
                    if server_manager.is_running("Travel Guide Agent"):
                        travel_tool = converter.a2a_to_langchain_tool("http://localhost:5005", "Travel Guide Agent")
                        tools.append(travel_tool)
                    
                    context["a2a_langchain_tools"] = tools
                    
                    if tools:
                        print(f"LangChain tools created from {len(tools)} A2A agents successfully")
                    else:
                        print("No A2A agents are running. Please start them first.")
                        
                except Exception as e:
                    print(f"Error creating tools: {e}")
            
            elif choice == "8":
                if not context["a2a_langchain_tools"]:
                    print("Please create LangChain tools from A2A agents first")
                else:
                    print("\nTest LangChain Tools from A2A")
                    
                    for i, tool in enumerate(context["a2a_langchain_tools"]):
                        test_query = input(f"Enter query for {tool.name} (or 'skip' to skip): ").strip()
                        if test_query.lower() != 'skip':
                            try:
                                result = tool.run(test_query)
                                print(f"{tool.name} Result: {result}")
                            except Exception as e:
                                print(f"{tool.name} Error: {e}")
            
            elif choice == "9":
                try:
                    tools = []
                    
                    # Add MCP tools if available
                    mcp_manager = MCPToolsManager(groq_client)
                    if server_manager.is_running("Calculator MCP"):
                        calc_tool = mcp_manager.mcp_to_langchain_tool("http://localhost:5001", "calculator")
                        tools.append(calc_tool)
                    if server_manager.is_running("Text Analyzer MCP"):
                        text_analyzer_tool = mcp_manager.mcp_to_langchain_tool("http://localhost:5002", "text_analyzer")
                        tools.append(text_analyzer_tool)
                    
                    # Add LangChain tools
                    if server_manager.is_running("LangChain MCP"):
                        lc_converter = LangChainToMCPConverter(groq_client)
                        lc_tools = lc_converter.create_langchain_tools()
                        tools.extend(lc_tools)
                    
                    # Add A2A tools dynamically
                    a2a_converter = A2AToLangChainConverter(groq_client)
                    if server_manager.is_running("Groq LLM Agent"):
                        llm_tool = a2a_converter.a2a_to_langchain_tool("http://localhost:5004", "Groq LLM Agent")
                        tools.append(llm_tool)
                    if server_manager.is_running("Travel Guide Agent"):
                        travel_tool = a2a_converter.a2a_to_langchain_tool("http://localhost:5005", "Travel Guide Agent")
                        tools.append(travel_tool)
                    
                    if tools:
                        graph_integration = LangGraphIntegration(groq_client)
                        graph = graph_integration.create_simple_graph(tools)
                        context["langgraph"] = graph
                        print(f"LangGraph created with {len(tools)} integrated tools successfully")
                        print(f"Available tools: {[tool.name for tool in tools]}")
                    else:
                        print("No tools available. Please start some servers first.")
                        
                except Exception as e:
                    print(f"Error creating LangGraph: {e}")
            
            elif choice == "10":
                if not context["langgraph"]:
                    print("Please create LangGraph first")
                else:
                    query = input("Enter your query for LangGraph: ").strip()
                    try:
                        result = context["langgraph"].invoke({"messages": [query], "cycle_count": 0, "should_continue": True})
                        print("LangGraph Result:")
                        for i, msg in enumerate(result["messages"]):
                            print(f"Step {i+1}: {msg}")
                    except Exception as e:
                        print(f"LangGraph Error: {e}")
            
            elif choice == "11":
                print("\nDiscover Agents and Capabilities")
                try:
                    # Initialize client for Directory Agent
                    directory_client = A2AClient("http://localhost:5000")
                    list_task = create_task({"skill": "list_agents"})
                    
                    # Get and parse list of agents
                    list_response = directory_client.ask(list_task)
                    list_response_dict = json.loads(list_response)  # Parse JSON string to dict
                    list_response_text = list_response_dict["artifacts"][0]["parts"][0]["text"]
                    agents = json.loads(list_response_text)  # Parse the text field (JSON string) to list
                    
                    print("Available Agents:")
                    for agent in agents:
                        print(f"- {agent['name']}: {agent['description']} at {agent['url']}")
                        
                        # Get and parse capabilities for each agent
                        agent_client = A2AClient(agent['url'])
                        cap_task = create_task({"skill": "get_capabilities"})
                        capabilities_response = agent_client.ask(cap_task)
                        capabilities_response_dict = json.loads(capabilities_response)  # Parse JSON string to dict
                        capabilities_response_text = capabilities_response_dict["artifacts"][0]["parts"][0]["text"]
                        capabilities = json.loads(capabilities_response_text)  # Parse the text field to dict
                        print(f"  Capabilities: {capabilities['skills']}")
                except Exception as e:
                    print(f"Error discovering agents: {e}")
            
            elif choice == "12":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up servers
        for name in list(server_manager.servers.keys()):
            server_manager.stop_server(name)
        sys.exit(0)

if __name__ == "__main__":
    main()