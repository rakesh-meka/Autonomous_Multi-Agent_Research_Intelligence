"""
ACE - Autonomous Cognitive Engine
Core State Management (LangGraph-style)
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import uuid
import json
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCH = "research"
    SUMMARIZER = "summarizer"
    REPORTER = "reporter"


@dataclass
class Task:
    id: str
    title: str
    description: str
    agent: AgentType
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    search_query: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "agent": self.agent.value,
            "priority": self.priority,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "result": self.result[:200] + "..." if len(self.result) > 200 else self.result,
            "search_query": self.search_query,
        }


@dataclass
class LogEntry:
    agent: str
    message: str
    level: str = "info"  # info | success | warning | error
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class EngineState:
    """
    Central state object passed through the LangGraph-style pipeline.
    All agents read from and write to this state.
    """
    # Input
    query: str = ""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Planning
    analysis: str = ""
    tasks: List[Task] = field(default_factory=list)

    # Memory / VFS
    memory: Dict[str, str] = field(default_factory=dict)       # task_title -> content
    vfs: Dict[str, Dict] = field(default_factory=dict)          # filename -> {content, metadata}

    # Execution tracking
    current_task_id: Optional[str] = None
    active_agent: Optional[str] = None
    logs: List[LogEntry] = field(default_factory=list)

    # Output
    final_report: str = ""
    status: str = "idle"   # idle | planning | executing | reporting | complete | error
    error_message: str = ""

    # Callbacks (not serialized)
    on_update: Optional[Callable] = field(default=None, repr=False)

    def log(self, agent: str, message: str, level: str = "info"):
        entry = LogEntry(agent=agent, message=message, level=level)
        self.logs.append(entry)
        if self.on_update:
            self.on_update(self)

    def update_task_status(self, task_id: str, status: TaskStatus, result: str = "", error: str = ""):
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                if result:
                    task.result = result
                if error:
                    task.error = error
                if status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.now().isoformat()
        if self.on_update:
            self.on_update(self)

    def write_vfs(self, filename: str, content: str, metadata: dict = None):
        self.vfs[filename] = {
            "content": content,
            "size": len(content),
            "updated": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        if self.on_update:
            self.on_update(self)

    def read_vfs(self, filename: str) -> Optional[str]:
        return self.vfs.get(filename, {}).get("content")

    def get_memory_context(self, max_items: int = 5, max_chars: int = 2000) -> str:
        """Get recent memory entries as context string."""
        items = list(self.memory.items())[-max_items:]
        parts = []
        total = 0
        for title, content in items:
            snippet = content[:max_chars // len(items)] if items else content
            parts.append(f"[{title}]:\n{snippet}")
            total += len(snippet)
        return "\n\n".join(parts)

    def get_all_memory(self) -> str:
        return "\n\n---\n\n".join(
            f"## {title}\n\n{content}" for title, content in self.memory.items()
        )

    @property
    def completed_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    @property
    def pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        return len(self.completed_tasks) / len(self.tasks)

    def to_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "query": self.query[:100],
            "status": self.status,
            "tasks": [t.to_dict() for t in self.tasks],
            "vfs_files": list(self.vfs.keys()),
            "memory_keys": list(self.memory.keys()),
            "progress": f"{self.progress:.0%}",
            "log_count": len(self.logs),
        }
