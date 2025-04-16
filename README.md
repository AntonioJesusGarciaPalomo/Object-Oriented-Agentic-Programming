# OOAP SDK: Object Oriented Agentic Programming

[![PyPI version](https://img.shields.io/pypi/v/ooap-sdk.svg)](https://pypi.org/project/ooap-sdk/)
[![Python Version](https://img.shields.io/pypi/pyversions/ooap-sdk.svg)](https://pypi.org/project/ooap-sdk/)
[![License](https://img.shields.io/github/license/AntonioJesusGarciaPalomo/Object-Oriented-Agentic-Programming.svg)](https://github.com/AntonioJesusGarciaPalomo/Object-Oriented-Agentic-Programming/blob/main/LICENSE)

**OOAP SDK** introduces a novel paradigm: **Object Oriented Agentic Programming**. This framework combines object-oriented programming principles with LLM-based AI agents, enabling the creation of robust, maintainable, and scalable multi-agent systems.

## 🚀 Vision and Overview

Traditional agent-based programming lacks the formal structures and guarantees that make OOP powerful. OOAP addresses this by treating agents as first-class objects with:

- **Encapsulation**: Internal state, memory, and context
- **Defined Interface**: Controlled exposed methods
- **Inheritance**: Shared capabilities between agent types
- **Polymorphism**: Different agents implementing the same interfaces
- **Validation**: Guarantees on inputs and outputs
- **Composition**: Building complex systems from simpler agents

OOAP integrates seamlessly with the Model Context Protocol (MCP), allowing agents to access resources, tools, and context in a standardized way.

## 📦 Installation

```bash
pip install ooap-sdk
```

Or install directly from the repository:

```bash
pip install git+https://github.com/AntonioJesusGarciaPalomo/Object-Oriented-Agentic-Programming.git
```

## 🧩 Architecture

OOAP SDK implements the following layers:

1. **Agent Core**: Base classes defining fundamental agent capabilities
2. **Memory Systems**: Short-term and long-term memory implementations
3. **LLM Integration**: Interfaces to language models via MCP and other APIs
4. **Validation**: Type system and validation based on Pydantic
5. **Orchestration**: Tools for coordinating multi-agent systems

### Agent Class Diagram

```
┌─────────────────┐       ┌─────────────────┐
│ Agent<T>        │       │ AgentEnvironment │
├─────────────────┤       ├─────────────────┤
│ id              │       │ variables       │
│ name            │◆─────▶│ tools           │
│ role            │       │ last_output     │
│ initial_context │       └─────────────────┘
│ short_term_mem  │              ▲
│ long_term_mem   │              │
│ environment     │              │
├─────────────────┤       ┌─────────────────┐
│ think()         │       │ ShortTermMemory │
│ _build_context()│       ├─────────────────┤
│ validate_output()│      │ messages        │
└─────────────────┘       │ capacity        │
        ▲                 ├─────────────────┤
        │                 │ add()           │
        │                 │ get_context()   │
┌─────────────────┐       └─────────────────┘
│ SpecializedAgent│              ▲
├─────────────────┤              │
│ specific_attr   │              │
├─────────────────┤       ┌─────────────────┐
│ specific_method()│      │ LongTermMemory  │
└─────────────────┘       ├─────────────────┤
                          │ entries         │
                          ├─────────────────┤
                          │ add()           │
                          │ get()           │
                          │ search()        │
                          └─────────────────┘
```

## 💡 Key Features

- **Declarative agent definitions** as Python classes
- **Internal state management** with short and long-term memory
- **Model Context Protocol (MCP) integration** for resources and tools access
- **Input and output validation** with Pydantic
- **Inter-agent communication** through well-defined interfaces
- **Support for agent inheritance and polymorphism**
- **Multi-agent system orchestration**
- **Standardized LLM interaction** patterns

## 🔍 Basic Usage

```python
from ooap import Agent, AgentMeta
from ooap.memory import ShortTermMemory, LongTermMemory
from pydantic import BaseModel, Field
from typing import List

# Define a model for the agent's output
class AnalysisResult(BaseModel):
    summary: str = Field(..., description="Summary of the analysis")
    key_points: List[str] = Field(..., description="Key points identified")
    
# Create an agent with AgentMeta metaclass for automatic validation
class AnalystAgent(Agent[AnalysisResult], metaclass=AgentMeta):
    OutputModel = AnalysisResult  # Model for validation
    
    def __init__(self, name: str, expertise: str):
        super().__init__(
            name=name,
            role=f"Analyst specialized in {expertise}",
            initial_context=f"You are an expert in {expertise} analysis."
        )
        self.expertise = expertise
    
    async def think(self, input_text: str) -> AnalysisResult:
        """Process input text and generate a structured analysis"""
        # The agent accesses its internal memory and context
        self.short_term_memory.add("user", input_text)
        
        # Process input using LLM (simplified here)
        result = await self._process_with_llm(input_text)
        
        # OOAP automatically validates result against AnalysisResult
        return result
    
    def search_references(self, query: str) -> List[str]:
        """Method exposed as a tool for the agent"""
        # The agent can call this method
        return [f"Reference about '{query}'", "Another relevant reference"]
```

## 🔮 Advanced Examples

### Multi-Agent System

```python
from ooap import AgentSystem
from ooap.agents import ResearchAgent, AnalystAgent, SummaryAgent

# Create a system with multiple specialized agents
system = AgentSystem()

# Register agents with different roles
system.register_agent(ResearchAgent("Researcher", research_focus="market data"))
system.register_agent(AnalystAgent("Analyst", expertise="financial trends"))
system.register_agent(SummaryAgent("Summarizer", style="concise"))

# Process a complex task requiring collaboration
result = await system.process_task(
    task="Analyze the potential impact of interest rates on the tech market",
    primary_agent="Analyst",  # Main agent for the task
    supporting_agents=["Researcher"]  # Agents providing complementary information
)

# Generate a final summary
summary = await system.agents["Summarizer"].think(str(result))
print(summary)
```

### Agent Inheritance

```python
from ooap import ContentAgent

# Base class for content agents
class MarketingAgent(ContentAgent):
    """Base agent for marketing"""
    def __init__(self, name: str, brand_guidelines: str):
        super().__init__(name=name, role="Marketing Specialist")
        self.brand_guidelines = brand_guidelines

# Specialized subclasses
class SocialMediaAgent(MarketingAgent):
    """Social media specialist"""
    def __init__(self, name: str, brand_guidelines: str, platform: str):
        super().__init__(name=name, brand_guidelines=brand_guidelines)
        self.platform = platform
        self.role = f"Social Media Specialist for {platform}"
    
    # Specialized implementation of a common method 
    def format_content(self, content: str) -> str:
        if self.platform == "Twitter":
            return content[:280]  # Character limit
        elif self.platform == "Instagram":
            return f"{content}\n\n#branded #campaign #{self.name}"
        else:
            return content
```

## 🧠 LLM Integration

OOAP SDK provides integration with different models through:

### MCP (Model Context Protocol)

```python
from ooap import MCPPoweredAgent
from ooap.llm import MCPProvider

# Create an agent using MCP to communicate with LLMs
agent = MCPPoweredAgent(
    name="DataAnalyst",
    role="Financial Analyst",
    llm_provider=MCPProvider(
        server_path="path/to/mcp_server.py",
        model_preferences={"model": "claude-3-opus-20240229"}
    )
)

# The agent will use MCP to access the LLM
result = await agent.think("Analyze these quarterly financial results")
```

### Other Providers

```python
from ooap.llm import OpenAIProvider, AnthropicProvider, LocalLLMProvider

# OOAP supports multiple LLM providers
openai_agent = MCPPoweredAgent(
    name="OpenAIAgent",
    llm_provider=OpenAIProvider(model="gpt-4")
)

anthropic_agent = MCPPoweredAgent(
    name="AnthropicAgent",
    llm_provider=AnthropicProvider(model="claude-3-sonnet")
)

local_agent = MCPPoweredAgent(
    name="LocalAgent",
    llm_provider=LocalLLMProvider(model_path="/path/to/local/model")
)
```

## 📊 Memory Systems

OOAP provides various memory implementations:

```python
from ooap.memory import VectorMemory, SQLiteMemory, RedisMemory

# Vector memory for semantic search
agent_with_vector_memory = AnalystAgent(
    name="VectorMemoryAgent",
    expertise="market analysis",
    long_term_memory=VectorMemory(embedding_model="text-embedding-3-small")
)

# Persistent memory with SQLite
agent_with_db = AnalystAgent(
    name="PersistentAgent",
    expertise="historical data",
    long_term_memory=SQLiteMemory(db_path="agent_memory.db")
)

# Distributed memory with Redis
distributed_agent = AnalystAgent(
    name="DistributedAgent",
    expertise="real-time analysis",
    long_term_memory=RedisMemory(redis_url="redis://localhost:6379")
)
```

## 🔄 Core Classes and Components

### Agent Base Class

The foundation of OOAP is the `Agent` class:

```python
class Agent(Generic[T]):
    """Base class for all OOAP agents"""
    
    def __init__(
        self, 
        name: str,
        role: str,
        initial_context: str = "",
        mcp_server: Optional[FastMCP] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.initial_context = initial_context
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.environment = AgentEnvironment()
        self.mcp = mcp_server or FastMCP(f"Agent-{name}")
```

### AgentMeta Metaclass

The metaclass that handles validation and type checking:

```python
class AgentMeta(type):
    """Metaclass for agent classes that sets up automatic validation"""
    
    def __new__(mcs, name, bases, attrs):
        # Check if an output model is defined
        output_model = attrs.get('OutputModel')
        
        # Wrap methods that need validation
        for attr_name, attr in attrs.items():
            if callable(attr) and not attr_name.startswith('_'):
                original_method = attr
                
                def create_validated_method(method, output_type):
                    async def validated_method(self, *args, **kwargs):
                        result = await method(self, *args, **kwargs)
                        # If there's an output model, validate
                        if output_type and not isinstance(result, output_type):
                            try:
                                # Try to convert/validate
                                result = output_type(**result) if isinstance(result, dict) else output_type(result)
                            except Exception as e:
                                raise TypeError(f"Output validation failed: {e}")
                        return result
                    return validated_method
                
                if output_model:
                    attrs[attr_name] = create_validated_method(original_method, output_model)
        
        return super().__new__(mcs, name, bases, attrs)
```

## 🚧 Project Structure

```
ooap-sdk/
├── ooap/
│   ├── __init__.py
│   ├── agent.py           # Core Agent class
│   ├── agent_meta.py      # AgentMeta metaclass
│   ├── agent_system.py    # Multi-agent orchestration
│   ├── environment.py     # Agent environment
│   ├── memory/            # Memory implementations
│   │   ├── __init__.py
│   │   ├── short_term.py
│   │   ├── long_term.py
│   │   ├── vector.py
│   │   ├── sqlite.py
│   │   └── redis.py
│   ├── llm/               # LLM integrations
│   │   ├── __init__.py
│   │   ├── mcp.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── local.py
│   ├── agents/            # Pre-built agent types
│   │   ├── __init__.py
│   │   ├── analyst.py
│   │   ├── researcher.py
│   │   ├── content.py
│   │   └── summary.py
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── validation.py
│       └── serialization.py
├── examples/              # Example implementations
│   ├── basic_agent.py
│   ├── multi_agent.py
│   └── specialized_agents.py
├── tests/                 # Unit tests
├── pyproject.toml         # Project configuration
├── setup.py               # Package setup
├── LICENSE                # MIT License
└── README.md              # This file
```

## 🛠️ Contributing

Contributions are welcome! Please read through our contribution guidelines before submitting a PR.

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ooap-sdk.git
cd ooap-sdk

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the MCP community for providing the infrastructure for LLM integration
- To all contributors and testers who make this project possible
- The Python community for the amazing tools and libraries that make this project possible

## 📚 Further Reading

- [Model Context Protocol Documentation](https://modelcontextprotocol.io)
- [Agent-Based Software Engineering](https://en.wikipedia.org/wiki/Agent-based_model)
- [Object-Oriented Programming Principles](https://en.wikipedia.org/wiki/Object-oriented_programming)