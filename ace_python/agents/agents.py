"""
ACE - Specialized Agent Implementations
Each agent is a callable that takes and returns EngineState.
"""
import json
from typing import Optional
from core.state import EngineState, TaskStatus, AgentType, Task
from core.llm_client import LLMClient
import uuid


# ─────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, llm: LLMClient, agent_type: AgentType):
        self.llm = llm
        self.agent_type = agent_type
        self.name = agent_type.value

    def log(self, state: EngineState, message: str, level: str = "info"):
        state.log(self.name, message, level)


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorAgent(BaseAgent):
    """
    Oversees the entire workflow.
    - Receives the initial query
    - Delegates to Planner
    - Monitors task completion
    - Triggers final reporting
    """

    SYSTEM_PROMPT = """You are the Supervisor Agent of an Autonomous Cognitive Engine.
Your role is to analyze queries, oversee research workflows, and ensure quality outcomes.
You coordinate between specialized sub-agents: Research, Summarizer, and Reporter.
Think strategically about what information is needed and how to best structure the investigation."""

    def __init__(self, llm: LLMClient):
        super().__init__(llm, AgentType.SUPERVISOR)

    def analyze_query(self, state: EngineState) -> EngineState:
        """Initial query analysis and delegation planning."""
        state.status = "planning"
        self.log(state, f"Analyzing query: '{state.query[:80]}...'", "info")

        analysis_prompt = f"""Analyze this research query and provide a brief strategic overview:

Query: {state.query}

Provide:
1. Core research objectives (2-3 sentences)
2. Key knowledge domains required
3. Estimated complexity (low/medium/high)
4. Recommended approach

Keep it concise and actionable."""

        analysis = self.llm.complete(analysis_prompt, self.SYSTEM_PROMPT)
        state.analysis = analysis
        self.log(state, "Query analysis complete", "success")
        state.log("supervisor", analysis[:200] + "...", "info")
        return state

    def should_continue(self, state: EngineState) -> str:
        """Routing condition: check if all tasks are done."""
        pending = [t for t in state.tasks if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)]
        if not pending:
            return "report"
        return "execute"


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Breaks the query into actionable subtasks with agent assignments.
    Creates the TODO list with priorities and dependencies.
    """

    SYSTEM_PROMPT = """You are a Planner Agent. Your job is to decompose complex research queries
into specific, actionable subtasks. Assign each task to the appropriate specialized agent:
- research: for gathering information, facts, data, and analysis
- summarizer: for synthesizing multiple research findings into coherent summaries
- reporter: for generating structured report sections and final outputs

Tasks should be ordered logically with clear dependencies."""

    SCHEMA = """{
  "analysis": "string - brief description of research strategy",
  "tasks": [
    {
      "id": "task_N",
      "title": "short task name",
      "description": "detailed description of exactly what to research/do",
      "agent": "research|summarizer|reporter",
      "priority": 1,
      "dependencies": ["task_ids"],
      "search_query": "specific search query for research tasks, null otherwise"
    }
  ]
}"""

    def __init__(self, llm: LLMClient):
        super().__init__(llm, AgentType.PLANNER)

    def create_plan(self, state: EngineState) -> EngineState:
        """Generate the task plan from the query."""
        self.log(state, "Creating detailed execution plan...", "info")

        prompt = f"""Create a comprehensive research plan for:

Query: {state.query}

Supervisor Analysis: {state.analysis}

Break this into 6-9 specific tasks. Include:
- 3-4 research tasks covering different aspects
- 1-2 summarization tasks to synthesize findings  
- 1-2 report generation tasks for structured output
- 1 final report compilation task (agent: reporter)

Order them logically. Research tasks should come before summarization."""

        plan = self.llm.complete_json(prompt, self.SYSTEM_PROMPT, self.SCHEMA)

        if not plan or "tasks" not in plan:
            self.log(state, "Planning failed, using fallback plan", "warning")
            plan = self._fallback_plan(state.query)

        state.tasks = [
            Task(
                id=t.get("id", f"task_{i}"),
                title=t["title"],
                description=t["description"],
                agent=AgentType(t.get("agent", "research")),
                priority=t.get("priority", i + 1),
                dependencies=t.get("dependencies", []),
                search_query=t.get("search_query"),
            )
            for i, t in enumerate(plan["tasks"])
        ]

        state.write_vfs("00_plan.json", json.dumps(plan, indent=2))
        self.log(state, f"Plan created: {len(state.tasks)} tasks", "success")

        for task in state.tasks:
            self.log(state, f"  [{task.agent.value}] {task.title}", "info")

        return state

    def _fallback_plan(self, query: str) -> dict:
        return {
            "analysis": "Fallback research plan",
            "tasks": [
                {"id": "task_1", "title": "Core Research", "description": f"Research the main aspects of: {query}", "agent": "research", "priority": 1, "dependencies": [], "search_query": query},
                {"id": "task_2", "title": "Deep Analysis", "description": f"Analyze implications and context of: {query}", "agent": "research", "priority": 2, "dependencies": ["task_1"], "search_query": None},
                {"id": "task_3", "title": "Synthesis", "description": "Synthesize all research findings", "agent": "summarizer", "priority": 3, "dependencies": ["task_1", "task_2"], "search_query": None},
                {"id": "task_4", "title": "Final Report", "description": "Generate comprehensive final report", "agent": "reporter", "priority": 4, "dependencies": ["task_3"], "search_query": None},
            ]
        }


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """
    Conducts deep research on specific topics.
    Uses web search tools and internal knowledge.
    Stores findings in VFS and memory.
    """

    SYSTEM_PROMPT = """You are an expert Research Agent with comprehensive knowledge across all domains.
Conduct thorough, rigorous research and provide detailed, accurate, well-structured findings.
Include:
- Specific facts, statistics, and data points
- Historical context and current state
- Expert perspectives and key debates
- Real-world examples and case studies
- Emerging trends and future directions

Be comprehensive and authoritative. Structure your response clearly with subheadings."""

    def __init__(self, llm: LLMClient, web_search_tool=None):
        super().__init__(llm, AgentType.RESEARCH)
        self.web_search = web_search_tool

    def execute(self, state: EngineState, task: Task) -> str:
        """Execute a research task and return findings."""
        self.log(state, f"Researching: {task.title}", "info")

        context = state.get_memory_context(max_items=3, max_chars=1500)

        # Try web search if available
        web_results = ""
        if self.web_search and task.search_query:
            try:
                self.log(state, f"Web search: '{task.search_query}'", "info")
                results = self.web_search.run(task.search_query)
                web_results = f"\n\nWeb Search Results:\n{results[:2000]}"
            except Exception as e:
                self.log(state, f"Web search failed: {e}", "warning")

        prompt = f"""Research Task: {task.title}

Detailed Objective: {task.description}

Original Query: {state.query}

{f"Prior Research Context:{chr(10)}{context}" if context else ""}
{web_results}

Provide comprehensive research findings. Include:
1. Overview and key concepts
2. Detailed analysis with specific facts and data
3. Current state and recent developments  
4. Key challenges, debates, or controversies
5. Expert consensus and divergent views
6. Implications and future outlook

Be thorough, specific, and well-organized."""

        result = self.llm.complete(prompt, self.SYSTEM_PROMPT)
        self.log(state, f"Research complete: {len(result)} chars", "success")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZATION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SummarizationAgent(BaseAgent):
    """
    Synthesizes research findings into coherent summaries.
    Identifies key themes, patterns, and insights across multiple sources.
    """

    SYSTEM_PROMPT = """You are an expert Summarization and Synthesis Agent.
You excel at:
- Identifying key themes and patterns across multiple research findings
- Drawing meaningful connections between different aspects
- Extracting the most important and actionable insights
- Creating clear, structured summaries that preserve nuance
- Highlighting consensus views and important disagreements

Your summaries are comprehensive yet concise, structured yet readable."""

    def __init__(self, llm: LLMClient):
        super().__init__(llm, AgentType.SUMMARIZER)

    def execute(self, state: EngineState, task: Task) -> str:
        """Execute a summarization task."""
        self.log(state, f"Synthesizing: {task.title}", "info")

        all_memory = state.get_all_memory()

        prompt = f"""Summarization Task: {task.title}

Objective: {task.description}

Original Query: {state.query}

All Research Findings:
{all_memory}

Create a comprehensive synthesis that:
1. Identifies the 5-7 most important themes
2. Shows how different findings connect and support each other
3. Highlights the most significant insights
4. Notes any contradictions or areas of uncertainty
5. Draws out key conclusions supported by the evidence

Structure with clear headings and be specific."""

        result = self.llm.complete(prompt, self.SYSTEM_PROMPT)
        self.log(state, "Synthesis complete", "success")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORTER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ReporterAgent(BaseAgent):
    """
    Generates professional structured reports.
    Creates both section reports and final comprehensive output.
    """

    SYSTEM_PROMPT = """You are an expert Report Generation Agent specializing in professional research reports.
You create clear, authoritative, well-structured documents that:
- Lead with an executive summary that captures key value
- Present findings in logical, hierarchical order
- Support claims with specific evidence and data
- Provide actionable recommendations
- Use professional markdown formatting with clear headers

Your reports are suitable for executive audiences and technical experts alike."""

    def __init__(self, llm: LLMClient):
        super().__init__(llm, AgentType.REPORTER)

    def execute(self, state: EngineState, task: Task) -> str:
        """Execute a report generation task."""
        self.log(state, f"Generating report: {task.title}", "info")

        all_memory = state.get_all_memory()

        prompt = f"""Report Generation Task: {task.title}

Objective: {task.description}

Original Query: {state.query}

Research Findings and Summaries:
{all_memory}

Generate a professional report section with:
- Clear section headers
- Specific findings with supporting evidence
- Data points and examples where relevant
- Key takeaways highlighted
- Professional, authoritative tone

Use markdown formatting."""

        result = self.llm.complete(prompt, self.SYSTEM_PROMPT)
        self.log(state, "Report section complete", "success")
        return result

    def generate_final_report(self, state: EngineState) -> str:
        """Compile the final comprehensive report."""
        self.log(state, "Compiling final comprehensive report...", "info")

        all_memory = state.get_all_memory()

        prompt = f"""Compile a comprehensive final research report.

Original Query: "{state.query}"

All Research, Analysis, and Summaries:
{all_memory}

Generate a complete, professional research report with these sections:

# [Descriptive Report Title]

## Executive Summary
[3-5 sentence overview of key findings and conclusions]

## Background & Context
[Necessary background to understand the topic]

## Key Findings
[Most important discoveries, organized by theme with specific details]

## Detailed Analysis
[In-depth examination of each major aspect, with evidence]

## Synthesis & Insights
[How findings connect, patterns observed, what they mean]

## Conclusions
[Clear, well-supported conclusions from the research]

## Recommendations & Next Steps
[Actionable recommendations based on findings]

## Appendix: Research Notes
[Brief summary of methodology and sources]

Make this comprehensive, professional, and genuinely valuable. Use specific facts and data throughout."""

        report = self.llm.complete(prompt, self.SYSTEM_PROMPT)
        self.log(state, f"Final report compiled: {len(report)} chars", "success")
        return report
