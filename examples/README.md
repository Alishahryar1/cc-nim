# Free Claude Code Examples

This directory contains production-ready examples and demonstration projects using Free Claude Code (fcc-claude). Each example showcases real-world patterns and advanced features of the platform.

## Projects

### File Organizer Pro

A complete, production-ready file organization system that demonstrates all major Claude Code features in an integrated project.

**Features Demonstrated:**
- 🏗️ **Structured Output** - Pydantic models for request/response validation with type safety
- ⚙️ **Async Background Tasks** - Non-blocking task submission with real-time progress polling
- ☁️ **Cloud Sessions** - Ready for remote execution of heavy lifting tasks
- 🤖 **Custom Agents** - Framework for specialized task-specific agents
- 🔔 **Webhooks & Notifications** - Integration points for external services
- 🔌 **MCP Server Integration** - Connection patterns for external data sources

**Components:**
- `file_organizer_enhanced.py` - Core organizer with recursive scanning, date-based organization, and type categorization
- `file_organizer_pro_api.py` - FastAPI server providing REST endpoints for programmatic access
- `file_organizer_pro_dashboard.html` - Real-time web dashboard for monitoring and task creation
- `FILE_ORGANIZER_PRO_GUIDE.md` - Comprehensive setup and API reference guide
- `README.md` - Complete project summary and architecture documentation

**Quick Start:**

```bash
cd examples/file_organizer_pro

# Install dependencies
pip install -r requirements.txt

# Start the API server
python file_organizer_pro_api.py

# Open the dashboard
# Navigate to: file:///absolute/path/to/file_organizer_pro_dashboard.html
```

**Key Endpoints:**
- `POST /tasks` - Create organization task
- `GET /tasks/{id}` - Check task status
- `GET /stats` - View system statistics
- `POST /tasks/{id}/summary` - Get detailed results

**Learn More:** See `README.md` in this directory for complete project documentation and next steps.

---

## Adding Your Own Examples

When adding new examples to this directory:

1. **Create a subdirectory** for your example (e.g., `examples/my_example/`)
2. **Include a README.md** with:
   - Clear description of what the example demonstrates
   - Features showcased
   - Setup instructions
   - Usage examples
3. **Keep it self-contained** with all necessary files and requirements
4. **Add type hints and docstrings** following the project's Python standards
5. **Include tests** if the example is complex

---

## Integration with fcc-claude

All examples are designed to work seamlessly with the fcc-claude CLI:

```bash
# Ask Claude to enhance an example
uv run fcc-claude -p "Enhance the file organizer to support S3 bucket organization"

# Use cloud sessions for heavy lifting
uv run fcc-claude --cloud -p "Scale the organizer to handle 1TB datasets"

# Create custom agents
uv run fcc-claude --agent "file_analyzer" -p "Analyze file organization patterns"
```

---

## Learning Paths

### Path 1: Understanding Structured Output
Start with: `file_organizer_pro_api.py` - Study how Pydantic models are used for API contracts

### Path 2: Async Programming
Start with: `file_organizer_pro_api.py` - Review `BackgroundTasks` pattern and `TaskManager` class

### Path 3: Web Dashboards
Start with: `file_organizer_pro_dashboard.html` - See real-time client-side polling and UI updates

### Path 4: Production Architecture
Start with: `README.md` - Study the full system architecture and deployment considerations

---

**Total Line Count:** ~1,800 lines of production-ready code + comprehensive documentation
