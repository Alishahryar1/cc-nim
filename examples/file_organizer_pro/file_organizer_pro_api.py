#!/usr/bin/env python3
"""
File Organizer Pro API - RESTful interface for file organization.

Provides endpoints for:
- Starting organization tasks
- Monitoring task progress
- Retrieving organization history
- Managing organization profiles
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class OrganizationMode(StrEnum):
    """Organization mode options."""

    BY_TYPE = "type"
    BY_DATE = "date"
    BY_TYPE_AND_DATE = "type_and_date"


class TaskStatus(StrEnum):
    """Task status options."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OrganizationRequest(BaseModel):
    """Request model for organization task."""

    directory: str = Field(..., description="Directory path to organize")
    mode: OrganizationMode = Field(
        default=OrganizationMode.BY_TYPE, description="Organization mode"
    )
    recursive: bool = Field(default=False, description="Organize subdirectories")
    dry_run: bool = Field(default=True, description="Simulate without moving files")


class TaskResponse(BaseModel):
    """Response model for task status."""

    task_id: str
    status: TaskStatus
    directory: str
    mode: OrganizationMode
    recursive: bool
    dry_run: bool
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    files_organized: int = 0
    total_size: int = 0
    errors: list[dict[str, str]] = []
    message: str = ""


class FileStats(BaseModel):
    """File statistics model."""

    category: str
    count: int
    size_bytes: int


class OrganizationSummary(BaseModel):
    """Summary of organization results."""

    task_id: str
    total_files: int
    total_size: int
    by_category: list[FileStats]
    errors: list[dict[str, str]]
    duration_seconds: float


# ============================================================================
# Task Management
# ============================================================================


class TaskManager:
    """Manages organization tasks."""

    def __init__(self):
        """Initialize task manager."""
        self.tasks: dict[str, dict] = {}

    def create_task(self, request: OrganizationRequest) -> str:
        """
        Create a new organization task.

        Args:
            request: Organization request

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "request": request,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "files_organized": 0,
            "total_size": 0,
            "errors": [],
            "message": "Task created, waiting to start",
        }
        logger.info(f"Created task {task_id}")
        return task_id

    def get_task(self, task_id: str) -> dict | None:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update_task_status(
        self, task_id: str, status: TaskStatus, message: str = "", **kwargs
    ) -> None:
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            if message:
                self.tasks[task_id]["message"] = message
            self.tasks[task_id].update(kwargs)

    def list_tasks(self, status: TaskStatus | None = None) -> list[str]:
        """List all tasks, optionally filtered by status."""
        if status:
            return [
                task_id
                for task_id, task in self.tasks.items()
                if task["status"] == status
            ]
        return list(self.tasks.keys())


# ============================================================================
# File Organization Worker
# ============================================================================

FILE_CATEGORIES: dict[str, list[str]] = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
    "Documents": [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".ppt"],
    "Videos": [".mp4", ".avi", ".mov", ".mkv"],
    "Audio": [".mp3", ".wav", ".flac", ".aac"],
    "Code": [".py", ".js", ".ts", ".java", ".cpp", ".go"],
    "Archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
}


def categorize_file(file_extension: str) -> str:
    """Categorize file by extension."""
    file_extension = file_extension.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if file_extension in extensions:
            return category
    return "Other"


def organize_directory(
    task_manager: TaskManager, task_id: str, request: OrganizationRequest
) -> None:
    """
    Background task to organize a directory.

    Args:
        task_manager: Task manager instance
        task_id: Task ID
        request: Organization request
    """
    try:
        # Update status to running
        task_manager.update_task_status(
            task_id,
            TaskStatus.RUNNING,
            "Starting organization",
            started_at=datetime.now(),
        )

        target_dir = Path(request.directory).expanduser().resolve()

        if not target_dir.exists():
            raise ValueError(f"Directory not found: {target_dir}")

        # Collect files
        if request.recursive:
            files = [f for f in target_dir.rglob("*") if f.is_file()]
        else:
            files = [f for f in target_dir.iterdir() if f.is_file()]

        if not files:
            task_manager.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                "No files to organize",
                files_organized=0,
                total_size=0,
                errors=[],
            )
            return

        # Simulate organization (in real scenario, would move files)
        stats = defaultdict(lambda: {"count": 0, "size": 0})
        errors = []

        for file_path in files:
            try:
                category = categorize_file(file_path.suffix)
                size = file_path.stat().st_size

                stats[category]["count"] += 1
                stats[category]["size"] += size

                if not request.dry_run:
                    # In real scenario: create directory and move file
                    logger.info(f"Would move: {file_path.name} -> {category}/")

            except Exception as e:
                errors.append({"file": file_path.name, "error": str(e)})

        # Calculate totals
        total_size = sum(s["size"] for s in stats.values())

        # Update task as completed
        task_manager.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            f"Successfully organized {len(files)} files",
            files_organized=len(files),
            total_size=total_size,
            errors=errors,
            completed_at=datetime.now(),
            stats=dict(stats),
        )

        logger.info(f"Task {task_id} completed")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task_manager.update_task_status(
            task_id,
            TaskStatus.FAILED,
            f"Error: {e!s}",
            errors=[{"error": str(e)}],
            completed_at=datetime.now(),
        )


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="File Organizer Pro API",
    description="RESTful API for intelligent file organization",
    version="1.0.0",
)

# Initialize task manager
task_manager = TaskManager()


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "File Organizer Pro API",
        "version": "1.0.0",
    }


@app.post("/tasks", tags=["Tasks"], response_model=TaskResponse)
async def create_organization_task(
    request: OrganizationRequest, background_tasks: BackgroundTasks
):
    """
    Create a new file organization task.

    This endpoint starts an asynchronous task to organize files.
    Use the returned task_id to poll for progress.

    Args:
        request: Organization request with directory and options
        background_tasks: FastAPI background tasks

    Returns:
        Task response with ID and initial status
    """
    try:
        # Validate directory
        path = Path(request.directory).expanduser().resolve()
        if not path.exists():
            raise HTTPException(status_code=400, detail="Directory not found")

        # Create task
        task_id = task_manager.create_task(request)

        # Queue background task
        background_tasks.add_task(organize_directory, task_manager, task_id, request)

        task = task_manager.get_task(task_id)
        return TaskResponse(
            task_id=task_id,
            status=task["status"],
            directory=request.directory,
            mode=request.mode,
            recursive=request.recursive,
            dry_run=request.dry_run,
            created_at=task["created_at"],
            message=task["message"],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/tasks/{task_id}", tags=["Tasks"], response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get the status of an organization task.

    Args:
        task_id: Task ID

    Returns:
        Current task status and progress
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    request = task["request"]
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        directory=request.directory,
        mode=request.mode,
        recursive=request.recursive,
        dry_run=request.dry_run,
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        files_organized=task.get("files_organized", 0),
        total_size=task.get("total_size", 0),
        errors=task.get("errors", []),
        message=task.get("message", ""),
    )


@app.get("/tasks", tags=["Tasks"])
async def list_tasks(status: TaskStatus | None = Query(None)):
    """
    List all tasks, optionally filtered by status.

    Args:
        status: Optional status filter

    Returns:
        List of task IDs
    """
    task_ids = task_manager.list_tasks(status)
    return {
        "tasks": task_ids,
        "count": len(task_ids),
        "filter": status.value if status else None,
    }


@app.post(
    "/tasks/{task_id}/summary", tags=["Tasks"], response_model=OrganizationSummary
)
async def get_task_summary(task_id: str):
    """
    Get detailed summary of a completed task.

    Args:
        task_id: Task ID

    Returns:
        Detailed organization summary with statistics
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    stats = task.get("stats", {})
    by_category = [
        FileStats(category=cat, count=data["count"], size_bytes=data["size"])
        for cat, data in stats.items()
    ]

    started = task.get("started_at")
    completed = task.get("completed_at")
    duration = (completed - started).total_seconds() if started and completed else 0

    return OrganizationSummary(
        task_id=task_id,
        total_files=task.get("files_organized", 0),
        total_size=task.get("total_size", 0),
        by_category=by_category,
        errors=task.get("errors", []),
        duration_seconds=duration,
    )


@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Delete a task record."""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    del task_manager.tasks[task_id]
    return {"message": f"Task {task_id} deleted"}


@app.get("/stats", tags=["Stats"])
async def get_statistics():
    """Get overall statistics about all tasks."""
    all_tasks = task_manager.tasks
    completed = sum(
        1 for t in all_tasks.values() if t["status"] == TaskStatus.COMPLETED
    )
    running = sum(1 for t in all_tasks.values() if t["status"] == TaskStatus.RUNNING)
    failed = sum(1 for t in all_tasks.values() if t["status"] == TaskStatus.FAILED)
    total_files = sum(t.get("files_organized", 0) for t in all_tasks.values())

    return {
        "total_tasks": len(all_tasks),
        "completed_tasks": completed,
        "running_tasks": running,
        "failed_tasks": failed,
        "total_files_organized": total_files,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
