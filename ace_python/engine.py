"""
ACE - Autonomous Cognitive Engine
Main orchestrator that builds and runs the LangGraph-style pipeline.
"""
import os
from typing import Callable, Optional
from core.state import EngineState, TaskStatus, AgentType
from core.graph import CognitiveGraph
from core.llm_client import LLMClient
from agents.agents import SupervisorAgent, PlannerAgent, ResearchAgent, SummarizationAgent, ReporterAgent

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


def build_engine(
    api_key: str,
    model: str = "claude-opus-4-5",
    enable_search: bool = True,
    on_update: Optional[Callable] = None,
) -> "CompiledCognitiveEngine":
    """
    Factory function: builds and compiles the full cognitive engine.
    
    Returns a CompiledCognitiveEngine ready to invoke.
    """
    # Initialize LLM
    llm = LLMClient(
        model=model,
        api_key=api_key,
        max_tokens=2048,
        temperature=0.3,
    )

    # Initialize web search if available
    search_tool = None
    if enable_search and SEARCH_AVAILABLE:
        try:
            search_tool = DuckDuckGoSearchRun()
        except Exception:
            pass

    # Initialize agents
    supervisor = SupervisorAgent(llm)
    planner = PlannerAgent(llm)
    researcher = ResearchAgent(llm, web_search_tool=search_tool)
    summarizer = SummarizationAgent(llm)
    reporter = ReporterAgent(llm)

    # ── BUILD THE GRAPH ────────────────────────────────────────────────────────

    graph = CognitiveGraph("ACE-Pipeline")

    # Node: Supervisor Analysis
    def node_supervisor(state: EngineState) -> EngineState:
        return supervisor.analyze_query(state)

    # Node: Planner
    def node_planner(state: EngineState) -> EngineState:
        return planner.create_plan(state)

    # Node: Task Executor (dispatches to correct sub-agent)
    def node_execute_tasks(state: EngineState) -> EngineState:
        pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
        if not pending:
            state.status = "reporting"
            return state

        # Sort by priority and check dependencies
        ready_tasks = []
        completed_ids = {t.id for t in state.tasks if t.status == TaskStatus.COMPLETED}

        for task in sorted(pending, key=lambda t: t.priority):
            deps_met = all(dep in completed_ids for dep in task.dependencies)
            if deps_met:
                ready_tasks.append(task)

        if not ready_tasks:
            # Dependencies not met, try next cycle
            state.log("supervisor", "Waiting for dependencies...", "warning")
            state.status = "reporting"  # Force completion to avoid infinite loop
            return state

        # Execute next ready task
        task = ready_tasks[0]
        state.current_task_id = task.id
        state.update_task_status(task.id, TaskStatus.IN_PROGRESS)

        try:
            if task.agent == AgentType.RESEARCH:
                result = researcher.execute(state, task)
            elif task.agent == AgentType.SUMMARIZER:
                result = summarizer.execute(state, task)
            elif task.agent == AgentType.REPORTER:
                result = reporter.execute(state, task)
            else:
                result = researcher.execute(state, task)  # Default to research

            # Store in memory and VFS
            state.memory[task.title] = result
            filename = f"{task.id}_{task.agent.value}.md"
            state.write_vfs(filename, result, {"task_id": task.id, "agent": task.agent.value})
            state.update_task_status(task.id, TaskStatus.COMPLETED, result=result)
            state.log("memory", f"Stored: {filename} ({len(result)} chars)", "info")

        except Exception as e:
            error_msg = f"Task failed: {str(e)}"
            state.update_task_status(task.id, TaskStatus.FAILED, error=error_msg)
            state.log(task.agent.value, error_msg, "error")

        return state

    # Node: Final Report Generation
    def node_generate_report(state: EngineState) -> EngineState:
        state.status = "reporting"
        final_report = reporter.generate_final_report(state)
        state.final_report = final_report
        state.write_vfs("FINAL_REPORT.md", final_report, {"type": "final_report"})
        state.status = "complete"
        state.current_task_id = None
        state.active_agent = None
        state.log("supervisor", f"✓ Research complete. Report: {len(final_report)} chars", "success")
        return state

    # ── WIRE THE GRAPH ─────────────────────────────────────────────────────────

    graph.add_node("supervisor", node_supervisor, "Query analysis")
    graph.add_node("planner", node_planner, "Task planning")
    graph.add_node("executor", node_execute_tasks, "Task execution")
    graph.add_node("reporter", node_generate_report, "Report generation")

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "planner")
    graph.add_edge("planner", "executor")

    # Conditional: keep executing until all done, then report
    def tasks_done_condition(state: EngineState) -> str:
        pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
        in_progress = [t for t in state.tasks if t.status == TaskStatus.IN_PROGRESS]
        if not pending and not in_progress:
            return "done"
        if state.status == "reporting":
            return "done"
        return "continue"

    graph.add_conditional_edge(
        "executor",
        tasks_done_condition,
        {"continue": "executor", "done": "reporter"}
    )

    graph.add_edge("reporter", "__end__")

    compiled = graph.compile()
    return CompiledCognitiveEngine(compiled, on_update)


class CompiledCognitiveEngine:
    """Ready-to-use cognitive engine."""

    def __init__(self, compiled_graph, on_update=None):
        self.graph = compiled_graph
        self.on_update = on_update

    def run(self, query: str) -> EngineState:
        """Run the full cognitive pipeline on a query."""
        state = EngineState(
            query=query,
            on_update=self.on_update,
        )
        state.status = "running"
        state.log("supervisor", f"Engine started for: {query[:80]}", "info")

        try:
            final_state = self.graph.invoke(state)
            return final_state
        except Exception as e:
            state.status = "error"
            state.error_message = str(e)
            state.log("supervisor", f"Engine error: {e}", "error")
            return state

    def print_visualization(self):
        """Print graph structure."""
        print(self.graph.graph.visualize())
