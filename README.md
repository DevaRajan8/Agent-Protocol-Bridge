# Agent Protocol Bridge - *Bridging the gap between AI agent frameworks* 
**Universal integration framework that seamlessly converts and bridges MCP tools, A2A agents, LangChain components, and LangGraph workflows into unified, interoperable AI systems.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 What is Agent Protocol Bridge?

Agent Protocol Bridge solves the fragmentation problem in the AI ecosystem by providing seamless interoperability between different agent frameworks. Instead of being locked into a single protocol, you can now:

- **Convert MCP tools** to LangChain tools and vice versa
- **Transform A2A agents** into LangChain-compatible tools
- **Integrate everything** into LangGraph workflows
- **Discover and manage** all agents through a unified directory

## ✨ Key Features

### 🔄 **Bidirectional Conversions**
- **MCP ↔ LangChain**: Convert Model Context Protocol tools to LangChain tools
- **A2A → LangChain**: Transform Agent-to-Agent protocols into LangChain-compatible tools
- **LangChain → MCP**: Package LangChain tools as MCP servers

### 🕸️ **LangGraph Integration**
- Unified workflow orchestration with all converted tools
- Smart routing and conditional execution
- Cycle detection and management

### 📋 **Agent Directory Service**
- Automatic agent registration and discovery
- Capability introspection
- Centralized agent management

### 🔧 **Production Ready**
- Multi-threaded server management
- Error handling and retry logic
- Comprehensive logging
- Interactive CLI interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Tools     │───►│                 │◄───│  LangChain      │
│   - Calculator  │    │     Agent       │    │  - Weather      │
│   - Analyzer    │    │   Protocol      │    │  - Translator   │
└─────────────────┘    │   Bridge        │    └─────────────────┘
                       │                 │
┌─────────────────┐    │ Server Manager  │    ┌─────────────────┐
│   A2A Agents    │───►│                 │───►│   LangGraph     │
│   - LLM Agent   │    │                 │    │   Unified       │
│   - Travel Bot  │    └─────────────────┘    │   Workflows     │
└─────────────────┘                           └─────────────────┘
                                                       │
                                                       ▼
                                               ┌─────────────────┐
                                               │   Directory     │
                                               │   Service       │
                                               │  (Discovery)    │
                                               └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install langgraph langchain python-a2a requests
export GROQ_API_KEY="your_groq_api_key_here"
```

### Installation

```bash
git clone https://github.com/DevaRajan8/agent-protocol-bridge
cd agent-protocol-bridge
pip install -r requirements.txt
```

### Run the Interactive CLI

```bash
python app2.py
```

## 🎯 Use Cases

### **Enterprise Integration**
- Unify existing AI tools across different frameworks
- Reduce vendor lock-in
- Leverage best tools from each ecosystem

### **Research & Development**
- Combine different agent architectures
- Rapid prototyping with mixed toolsets
- Comparative analysis of agent frameworks

### **Microservices Architecture**
- Deploy each agent type as independent services
- Scale components individually
- Maintain framework diversity

### **Legacy System Integration**
- Bridge older AI systems with modern frameworks
- Gradual migration strategies
- Maintain existing investments

## 🔧 Available Conversions

| From | To | Status |
|------|----|---------| 
| MCP Tools | LangChain Tools |
| LangChain Tools | MCP Server |
| A2A Agents | LangChain Tools |
| All Tools | LangGraph Workflow |
| Agent Discovery | Directory Service |

## 🏃‍♂️ CLI Menu Options

```
0. View Server Status
1. Start MCP Servers
2. Test MCP to LangChain Conversion
3. Start LangChain Tools as MCP Server
4. Test Converted LangChain Tools
5. Start A2A Agents
6. Test A2A Agents
7. Create LangChain Tools from A2A Agents
8. Test LangChain Tools from A2A
9. Create LangGraph with Tools
10. Test LangGraph
11. Discover Agents and Capabilities
12. Exit
```

## 🛠️ Configuration

### Environment Variables

```bash
export GROQ_API_KEY="your_groq_api_key"        # Required for LLM functionality
export LOG_LEVEL="INFO"                         # Optional: DEBUG, INFO, WARNING, ERROR
export DEFAULT_TIMEOUT="30"                     # Optional: API timeout in seconds
```

### Server Ports

- Directory Agent: `5000`
- Calculator MCP: `5001`
- Text Analyzer MCP: `5002`
- LangChain MCP: `5003`
- Groq LLM Agent: `5004`
- Travel Guide Agent: `5005`


---

### Contact 

[Devarajan S](mailto:devarajan8.official@gmail.com)
