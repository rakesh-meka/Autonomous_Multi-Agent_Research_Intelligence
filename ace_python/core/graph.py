"""
ACE - LangGraph-style Pipeline Graph
Defines nodes, edges, and the execution graph.
"""
from typing import Callable, Dict, List, Optional, Tuple
from core.state import EngineState, TaskStatus


class Node:
    """A single node in the cognitive graph."""

    def __init__(self, name: str, func: Callable[[EngineState], EngineState], description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def run(self, state: EngineState) -> EngineState:
        state.active_agent = self.name
        state.log(self.name, f"Node '{self.name}' activated", "info")
        result = self.func(state)
        return result


class ConditionalEdge:
    """Routes to different nodes based on state."""

    def __init__(self, condition: Callable[[EngineState], str], routes: Dict[str, str]):
        self.condition = condition
        self.routes = routes  # condition_result -> node_name

    def resolve(self, state: EngineState) -> str:
        result = self.condition(state)
        return self.routes.get(result, list(self.routes.values())[0])


class CognitiveGraph:
    """
    LangGraph-style directed graph for orchestrating the cognitive pipeline.
    
    Nodes represent agent operations.
    Edges define execution flow.
    Conditional edges enable dynamic routing.
    """

    def __init__(self, name: str = "ACE"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, str] = {}                         # node -> next_node
        self.conditional_edges: Dict[str, ConditionalEdge] = {}  # node -> ConditionalEdge
        self.entry_point: Optional[str] = None
        self.terminal_nodes: List[str] = []

    def add_node(self, name: str, func: Callable, description: str = "") -> "CognitiveGraph":
        self.nodes[name] = Node(name, func, description)
        return self

    def set_entry_point(self, node_name: str) -> "CognitiveGraph":
        self.entry_point = node_name
        return self

    def add_edge(self, from_node: str, to_node: str) -> "CognitiveGraph":
        if to_node == "__end__":
            self.terminal_nodes.append(from_node)
        else:
            self.edges[from_node] = to_node
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable[[EngineState], str],
        routes: Dict[str, str]
    ) -> "CognitiveGraph":
        self.conditional_edges[from_node] = ConditionalEdge(condition, routes)
        # Mark terminal routes
        for route in routes.values():
            if route == "__end__":
                self.terminal_nodes.append(from_node)
        return self

    def get_next_node(self, current_node: str, state: EngineState) -> Optional[str]:
        """Determine next node from current node and state."""
        # Check conditional edge first
        if current_node in self.conditional_edges:
            next_node = self.conditional_edges[current_node].resolve(state)
            if next_node == "__end__":
                return None
            return next_node

        # Check direct edge
        if current_node in self.edges:
            return self.edges[current_node]

        # Terminal node
        return None

    def compile(self) -> "CompiledGraph":
        """Compile the graph for execution."""
        if not self.entry_point:
            raise ValueError("Entry point not set")
        return CompiledGraph(self)

    def visualize(self) -> str:
        """Return ASCII representation of the graph."""
        lines = [f"Graph: {self.name}", "=" * 40]
        for name, node in self.nodes.items():
            marker = "→ " if name == self.entry_point else "  "
            lines.append(f"{marker}[{name}] {node.description}")

            if name in self.edges:
                lines.append(f"    └─→ {self.edges[name]}")
            elif name in self.conditional_edges:
                ce = self.conditional_edges[name]
                for condition, target in ce.routes.items():
                    lines.append(f"    ├─ if '{condition}' → {target}")
            elif name in self.terminal_nodes:
                lines.append("    └─→ END")
        return "\n".join(lines)


class CompiledGraph:
    """Executable compiled graph."""

    def __init__(self, graph: CognitiveGraph):
        self.graph = graph

    def invoke(self, state: EngineState) -> EngineState:
        """Execute the graph synchronously."""
        current = self.graph.entry_point
        max_iterations = 50  # Safety limit
        iterations = 0

        while current and iterations < max_iterations:
            if current not in self.graph.nodes:
                raise ValueError(f"Node '{current}' not found in graph")

            node = self.graph.nodes[current]
            state = node.run(state)

            if state.status == "error":
                break

            next_node = self.graph.get_next_node(current, state)
            current = next_node
            iterations += 1

        return state

    async def ainvoke(self, state: EngineState) -> EngineState:
        """Async execution wrapper."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, state)
