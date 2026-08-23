# File Organizer Pro - Complete Setup Guide

## 📦 Project Overview

**File Organizer Pro** is a production-ready system demonstrating advanced Claude Code features:

- ✅ Core file organizer with multiple organization modes
- ✅ RESTful API server (FastAPI) with async task handling
- ✅ Web dashboard for monitoring and control
- ✅ Background task processing
- ✅ Cloud session support for remote execution
- ✅ Structured JSON output and validation

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic python-multipart
```

### 2. Start the API Server

```bash
python /home/user/file_organizer_pro_api.py
```

Server starts at: `http://localhost:8000`

### 3. Open the Dashboard

Open this file in your browser:
```
file:///home/user/file_organizer_pro_dashboard.html
```

### 4. Create an Organization Task

Use the dashboard or API:

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/organize",
    "mode": "type_and_date",
    "recursive": true,
    "dry_run": true
  }'
```

---

## 🎯 Advanced Claude Code Features Used

### 1️⃣ **Structured Output (JSON Schemas)**

The API uses Pydantic models for request/response validation:

```python
class OrganizationRequest(BaseModel):
    directory: str
    mode: OrganizationMode
    recursive: bool
    dry_run: bool


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    files_organized: int
    total_size: int
```

**Use with Claude Code:**
```bash
uv run fcc-claude -p "Validate this JSON against a schema" \
  --json-schema '{"type":"object","properties":{"files":{"type":"number"}}}'
```

---

### 2️⃣ **Background Tasks & Async Operations**

The API server handles long-running tasks asynchronously:

```python
@app.post("/tasks")
async def create_organization_task(request, background_tasks):
    task_id = task_manager.create_task(request)
    
    # Queue background task
    background_tasks.add_task(organize_directory, ...)
    
    return {"task_id": task_id, "status": "pending"}
```

**Benefits:**
- Returns immediately with task_id
- Client polls `/tasks/{task_id}` for progress
- Can run multiple tasks concurrently

---

### 3️⃣ **Cloud Sessions**

Run heavy organization tasks in the cloud:

```bash
# Create cloud session for remote execution
uv run fcc-claude --cloud "Run file organizer on large directory" \
  -p "Organize ~/massive_downloads with recursive mode"

# Cloud session handles:
# - Large directory scans
# - Heavy file I/O operations
# - Long-running tasks without local resource drain
```

**Advanced: Using Remote Execution**

```bash
# Start a background agent in cloud
uv run fcc-claude --bg --cloud \
  -p "Monitor ~/Downloads and auto-organize daily"
```

---

### 4️⃣ **Custom Agents**

Create specialized agents for different tasks:

```bash
# Agent for dry-run analysis
uv run fcc-claude --agent "organizer_analyzer" \
  -p "Analyze what would happen if we organized ~/Downloads"

# Agent for optimization
uv run fcc-claude --agent "organizer_optimizer" \
  -p "Suggest the best organization mode for ~/Downloads based on file types"
```

---

### 5️⃣ **Webhooks & Notifications**

In production, add webhooks for task completion:

```python
async def notify_on_completion(task_id: str):
    task = task_manager.get_task(task_id)
    if task["status"] == "completed":
        # Send webhook
        requests.post(
            "https://your-webhook.com/organize-complete",
            json={"task_id": task_id, "files": task["files_organized"]},
        )
```

---

### 6️⃣ **MCP Server Integration**

Connect to external data sources:

```python
# Example: Store results in database via MCP
from mcp import connect_to_server


async def store_results(task_id: str):
    # Connect to MCP database server
    async with connect_to_server("postgresql") as db:
        task = task_manager.get_task(task_id)
        await db.execute(
            "INSERT INTO organization_tasks VALUES ($1, $2, $3)",
            task_id,
            task["files_organized"],
            task["total_size"],
        )
```

---

## 📊 API Endpoints Reference

### Create Organization Task
```bash
POST /tasks
Content-Type: application/json

{
  "directory": "/path/to/organize",
  "mode": "type_and_date",  # or "type", "date"
  "recursive": true,
  "dry_run": true
}

Response: { "task_id": "uuid", "status": "pending" }
```

### Get Task Status
```bash
GET /tasks/{task_id}

Response: {
  "task_id": "uuid",
  "status": "running",
  "files_organized": 42,
  "total_size": 524288000,
  "message": "Organizing files..."
}
```

### List All Tasks
```bash
GET /tasks?status=completed

Response: {
  "tasks": ["task_id_1", "task_id_2"],
  "count": 2,
  "filter": "completed"
}
```

### Get Task Summary
```bash
POST /tasks/{task_id}/summary

Response: {
  "task_id": "uuid",
  "total_files": 150,
  "total_size": 5242880000,
  "by_category": [
    { "category": "Documents", "count": 50, "size_bytes": 104857600 },
    { "category": "Images", "count": 75, "size_bytes": 2097152000 }
  ],
  "duration_seconds": 45.2
}
```

### Get Statistics
```bash
GET /stats

Response: {
  "total_tasks": 10,
  "completed_tasks": 8,
  "running_tasks": 1,
  "failed_tasks": 1,
  "total_files_organized": 1250
}
```

---

## 💡 Usage Examples

### Example 1: Organize with API and Monitor

```bash
# Start task
TASK_ID=$(curl -s -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "~/Downloads",
    "mode": "type_and_date",
    "recursive": true,
    "dry_run": true
  }' | jq -r '.task_id')

echo "Started task: $TASK_ID"

# Poll for completion
while true; do
  STATUS=$(curl -s http://localhost:8000/tasks/$TASK_ID)
  STATE=$(echo $STATUS | jq -r '.status')
  FILES=$(echo $STATUS | jq -r '.files_organized')
  
  echo "Status: $STATE, Files: $FILES"
  
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    break
  fi
  
  sleep 2
done

# Get summary
curl http://localhost:8000/tasks/$TASK_ID/summary | jq
```

### Example 2: Using Claude Code to Generate Reports

```bash
# Ask Claude to analyze task results
uv run fcc-claude -p "I have an organization task result with 250 files, 
2.5GB total, organized into 8 categories. Generate a summary report and 
recommendations for backup strategy."
```

### Example 3: Batch Processing with Cloud Sessions

```bash
# Create cloud session for multiple directories
uv run fcc-claude --cloud "Batch organize multiple directories" \
  -p "I need to organize these directories using the API:
  1. ~/Downloads (recursive, by type and date)
  2. ~/Documents (recursive, by type)
  3. ~/Pictures (by date only)
  
  For each, make dry-run requests first, then actual requests.
  Provide a combined report."
```

---

## 🔧 Production Deployment

### Using Docker

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY file_organizer_pro_api.py .
COPY file_organizer_enhanced.py .

EXPOSE 8000

CMD ["python", "file_organizer_pro_api.py"]
```

Build and run:
```bash
docker build -t file-organizer-pro .
docker run -p 8000:8000 -v /data:/data file-organizer-pro
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: file-organizer-pro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: file-organizer
  template:
    metadata:
      labels:
        app: file-organizer
    spec:
      containers:
      - name: api
        image: file-organizer-pro:latest
        ports:
        - containerPort: 8000
        env:
        - name: WORKERS
          value: "4"
```

---

## 📈 Performance Considerations

### For Large Directories

1. **Use Recursive Mode Wisely**
   ```bash
   # Fast: Non-recursive (only root files)
   curl ... -d '{"recursive": false}'
   
   # Slower: Recursive (includes subdirectories)
   curl ... -d '{"recursive": true}'
   ```

2. **Batch Processing**
   - Process large directories in chunks
   - Use dry-run to estimate time
   - Queue multiple tasks, don't wait for completion

3. **Cloud Sessions for Heavy Lifting**
   ```bash
   # Remote processing doesn't block local machine
   uv run fcc-claude --cloud --bg \
     -p "Organize 500GB archive"
   ```

---

## 🐛 Troubleshooting

### API Server Won't Start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
export PORT=8001
# Modify startup code accordingly
```

### Dashboard Can't Connect

- Ensure API server is running on `localhost:8000`
- Check browser console for CORS errors
- API should have CORS middleware in production:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Tasks Failing

Check API logs:
```bash
# Increase logging verbosity
export LOG_LEVEL=DEBUG
python file_organizer_pro_api.py
```

---

## 🎓 Learning Advanced Features

### With Claude Code

Use the CLI to explore advanced features:

```bash
# Learn structured output
uv run fcc-claude -p "Explain Pydantic models and how to use them for API validation"

# Learn async patterns
uv run fcc-claude -p "How do background tasks work in FastAPI? Show an example."

# Cloud sessions
uv run fcc-claude -p "What are cloud sessions in Claude Code? When should I use them?"

# Agents
uv run fcc-claude -p "How do I create custom agents in Claude Code?"
```

---

## 📚 Files in This Project

```
/home/user/
├── file_organizer_enhanced.py       # Core organizer (recursive, date-based)
├── file_organizer_pro_api.py        # FastAPI server with async tasks
├── file_organizer_pro_dashboard.html # Web UI for monitoring
└── FILE_ORGANIZER_PRO_GUIDE.md      # This file
```

---

## 🚀 Next Steps

1. **Start the API server** and explore the dashboard
2. **Create test tasks** with different organization modes
3. **Integrate with Claude Code** for analysis and reporting
4. **Deploy to cloud** using Docker or Kubernetes
5. **Add monitoring** with Prometheus/Grafana
6. **Implement webhooks** for notifications

---

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Use Claude Code to debug:
   ```bash
   uv run fcc-claude -p "Here's my error: [error]. How do I fix it?"
   ```
3. Review FastAPI documentation: https://fastapi.tiangolo.com

---

**Happy organizing! 🎉**
