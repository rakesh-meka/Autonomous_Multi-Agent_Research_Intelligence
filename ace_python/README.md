# ⬡ ACE — Autonomous Cognitive Engine

> A multi-agent deep research system with LangGraph-style orchestration, specialized AI agents, virtual file system, and Streamlit UI.

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│  Supervisor  │  ← Analyzes query, oversees workflow
└──────┬───────┘
       │
    ▼
┌──────────────┐
│   Planner    │  ← Decomposes into 6-9 subtasks with agent assignments
└──────┬───────┘
       │
    ▼
┌─────────────────────────────────────┐
│         LangGraph Pipeline          │
│                                     │
│  ┌──────────┐  ┌────────────────┐   │
│  │ Research │  │  Summarizer    │   │
│  │  Agent   │  │    Agent       │   │
│  └────┬─────┘  └───────┬────────┘   │
│       │                │            │
│       ▼                ▼            │
│  ┌─────────────────────────────┐    │
│  │    Virtual File System      │    │
│  │  (task_N_agent.md files)    │    │
│  └─────────────┬───────────────┘    │
│                │                    │
│    ▼                                │
│  ┌──────────────┐                   │
│  │   Reporter   │                   │
│  │    Agent     │                   │
│  └──────────────┘                   │
└──────────────┬──────────────────────┘
               │
    ▼
┌──────────────────┐
│   Final Report   │  ← Comprehensive structured output
└──────────────────┘
```

## Features

- **LangGraph-style Pipeline**: Custom state machine with typed nodes, edges, and conditional routing
- **6 Specialized Nodes**: Supervisor, Planner, Research, Summarizer, Reporter, Memory
- **Virtual File System**: Intermediate results stored and retrieved during execution
- **Memory Context**: Each agent has access to prior findings for coherent research
- **Web Search**: DuckDuckGo integration for real-time information (optional)
- **Streaming**: Live output streaming during agent execution
- **Streamlit UI**: Dark-themed, professional interface with real-time updates
- **Task Management**: Full TODO lifecycle (Pending → In Progress → Completed)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Or set directly:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Launch the App

```bash
streamlit run app.py
```

### 4. Run Programmatically

```python
from engine import build_engine

engine = build_engine(api_key="your_key")
state = engine.run("Analyze the future of quantum computing")

print(state.final_report)
print(f"Tasks completed: {len(state.completed_tasks)}")
print(f"VFS files: {list(state.vfs.keys())}")
```

## Project Structure

```
ace_python/
├── app.py                  # Streamlit UI
├── engine.py               # Graph builder & main orchestrator
├── requirements.txt
├── .env.example
│
├── core/
│   ├── state.py            # EngineState dataclass (LangGraph-style)
│   ├── graph.py            # CognitiveGraph, Node, ConditionalEdge classes
│   └── llm_client.py       # Anthropic/LangChain LLM wrapper
│
└── agents/
    └── agents.py           # All agent implementations
```

## Agent Details

### SupervisorAgent
- Analyzes incoming query for scope and complexity
- Provides strategic direction
- Routes tasks via conditional edges

### PlannerAgent  
- Uses LLM to decompose query into 6-9 subtasks
- Assigns each task to appropriate agent type
- Sets priorities and dependencies

### ResearchAgent
- Deep research using LLM knowledge + optional web search
- Maintains context from prior research tasks
- Stores results in VFS as `.md` files

### SummarizationAgent
- Synthesizes all research findings
- Identifies themes, patterns, and connections
- Creates structured summaries

### ReporterAgent
- Generates professional report sections
- Compiles final comprehensive report
- Formats with executive summary, findings, analysis, recommendations

## LangGraph Implementation

The system implements LangGraph concepts natively:

```python
graph = CognitiveGraph("ACE")

graph.add_node("supervisor", node_supervisor_func)
graph.add_node("planner", node_planner_func)
graph.add_node("executor", node_executor_func)
graph.add_node("reporter", node_reporter_func)

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "planner")
graph.add_edge("planner", "executor")

# Conditional routing: loop until all tasks done
graph.add_conditional_edge(
    "executor",
    tasks_done_condition,           # Returns "done" | "continue"
    {"continue": "executor", "done": "reporter"}
)

graph.add_edge("reporter", "__end__")
compiled = graph.compile()
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| Model | `claude-opus-4-5` | Claude model to use |
| Web Search | enabled | DuckDuckGo integration |

## Extending the System

### Adding a New Agent

```python
# 1. Add to AgentType enum (state.py)
class AgentType(str, Enum):
    MY_AGENT = "my_agent"

# 2. Create agent class (agents/agents.py)
class MyAgent(BaseAgent):
    SYSTEM_PROMPT = "You are..."
    
    def execute(self, state: EngineState, task: Task) -> str:
        result = self.llm.complete(task.description, self.SYSTEM_PROMPT)
        return result

# 3. Add node to graph (engine.py)
my_agent = MyAgent(llm)

def node_my_agent(state):
    # Find pending tasks for this agent and execute
    ...
    return state

graph.add_node("my_agent", node_my_agent)
```

### Using Real LangGraph

To use the actual LangGraph library instead of this native implementation:

```python
from langgraph.graph import StateGraph, END

builder = StateGraph(EngineState)
builder.add_node("supervisor", node_supervisor)
builder.set_entry_point("supervisor")
# ... etc
graph = builder.compile()
```

The `EngineState` dataclass is compatible with LangGraph's state management.
