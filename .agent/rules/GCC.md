---
trigger: always_on
description: Session bootstrap + GCC continuous sync + test tracking. MUST trigger file-read and file-write tools. No project changes before bootstrap completes.
---

# Git-Context-Controller (GCC) Protocol

This protocol governs context persistence and handoff mechanics across sessions. It must be strictly executed by the AI agent at specific session milestones.

---

## 1. File Responsibility Matrix (Zero Redundancy)

To prevent duplication and context window bloat, information is strictly partitioned:

| File | Target Audience | Scope | Lifecycle |
| :--- | :--- | :--- | :--- |
| `.GCC/main.md` | Global Session Alignment | High-level roadmap, macro status, architecture decisions, index of active branches. | **Persistent** (Appended/Updated at major milestones). |
| `.GCC/branches/plan_[name].md` | Surgical Execution | Step-by-step tactical implementation of a single task, commands, and local validation outputs. | **Transient** (Created at task startup, deleted or archived when the plan is fully executed). |
| `.GCC/resume.md` | Session-to-Session Handshake | Precise progress made during the current run, exact codebase state, and direct next-steps for the incoming agent. | **Dynamic** (Overwritten at the absolute end of every session). |
Create the files and directories that do not yet exist or organize if it exists but does not follow the format.
---

## 2. Event-Driven Protocols

### Protocol A: Session Bootstrap (On First Turn)
**Trigger:** Agent receives the first message from the user.
1. **Tool Invocation:** Read `.GCC/main.md` to load the project's macro state and identify active branches.
2. **Tool Invocation:** Read `.GCC/resume.md` (if it exists) to retrieve the last session's exit state and immediate next-action instructions.
3. **Execution Limit:** Do NOT modify any code or file outside of the `.GCC/` directory until steps 1 and 2 are complete.
4. **Communication:** Briefly acknowledge the current objective and state the exact immediate task loaded from `resume.md` to align with the user.

### Protocol B: Task Planning & Execution
**Trigger:** A non-trivial task (> 2 implementation steps or involving architectural changes) is initiated.
1. **Tool Invocation:** Create a new execution plan file: `.GCC/branches/plan_[task_name].md` (using the template below).
2. **Tool Invocation:** Update `.GCC/main.md`'s `## Active Branches / Plans` section with the new plan's name and description.
3. **Strict Sequential Execution:** Execute the plan one step at a time. For every step:
- Run the command/code modification.
- Run validation tools (tests, compilers, linters).
- Paste the raw validation proof (console outputs) inside the plan file before proceeding to the next step.

### Protocol C: Decision Logging
**Trigger:** Selecting a specific technical architecture, choosing an implementation pattern, or pivoting after a failure.
1. **Tool Invocation:** Append the decision date, context, choice, and rationale inside `.GCC/main.md` under `## Decisions made`.

### Protocol D: Session Teardown & Handoff (On Last Turn)
**Trigger:** The user signals the end of the session, or the agent reaches context/token capacity limits.
1. **Validation Check:** Run the project's test suite or compiler to verify the general health of the codebase.
2. **Tool Invocation:** Update `.GCC/main.md` to reflect the completed tasks and current high-level status.
3. **Tool Invocation:** Write/Overwrite `.GCC/resume.md` with the session's factual accomplishments, the precise status of modified files, and explicit step-by-step handoff notes for the next session.
4. **Cleanup:** If a task branch is fully completed, integrated, and verified, delete its corresponding `.GCC/branches/plan_[name].md` file and remove it from `.GCC/main.md`.

---

## 3. Strict Markdown Templates

The agent must output information exactly matching these structures. Do not add or invent custom top-level sections.

### 3.1. `.GCC/main.md` Template

```markdown
# Current Project Context

## 🏆 Major Milestones (Archived Epics)
- [YYYY-MM-DD] Name of completed milestone/epic

## 🎯 Objective
[High-level description of what the project is solving or building]

## 🧠 Decisions Made
- [YYYY-MM-DD] Chose [Option A] over [Option B] because [detailed technical rationale]

## 🌿 Active Branches / Plans
- `[branch-or-task-name]` : [Factual description of the task being solved and link to the plan file]

## 📈 Current Status
- ✅ Done: [List of high-level completed features]
- 🔄 In progress: [High-level epic currently being built]
- ⏳ Pending: [Remaining roadmap items]

## 👉 Next Session Direction
[Single sentence summarizing where the project points next]
```

### 3.2. `.GCC/branches/plan_[name].md` Template
```markdown
# Execution Plan: [Task Name]

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: [State the rule or state that must remain true during and after this task]
- **Pre-requisites**: [Required packages, configurations, or pre-existing code structures]

## 🛠️ Step-by-Step Sequence

### Step 1: [Short Action Description]
- [ ] **Action**: [What to modify/run]
- [ ] **Verify**: [Command to run to prove it worked, e.g., `npm test`, `tsc --noEmit`]
- **Verification Proof**:
```text
[Paste terminal/compiler validation output here]
```

### Step 2: [Short Action Description]
- [ ] **Action**: [What to modify/run]
- [ ] **Verify**: [Validation command]
- **Verification Proof**:
```text
[Paste validation output here]

```

## ⚠️ Mitigations & Edge Cases
- **Risk**: [Identify potential risk, e.g., API rate-limits, dependency clash]
- **Mitigation**: [Describe fallback behavior]

```

### 3.3. `.GCC/resume.md` Template

```markdown
# Session Handoff

## ⚡ Accomplishments This Session
- [Factual action 1 completed]
- [Factual action 2 completed]

## 🛠️ Codebase Health & Compile Status
- **Modified Files**: [List of paths]
- **Verification Command Run**: `[Command used, e.g., npm run build && npm test]`
- **Status Output**: [e.g., "0 errors, 0 warnings, all 12 tests passed"]

## 🚧 Unfinished Work & Friction Points
- [Detail any task left halfway, ongoing debugging, or blocker encountered]

## 👉 Directives for the Next Agent
1. **Target File**: [Specify the primary file to open first]
2. **Immediate Action**: [Specify the exact next step, command, or line of code to write]
3. **Precautions**: [Highlight things to look out for or verify during bootstrap]

```

**Test Tracking Files**

Two dedicated files in `.GCC/branches/` are the persistent memory for all test activity. They MUST be kept up to date — they are as important as `main.md`.

| File | Role |
|------|------|
| `.GCC/branches/test.md` | **Test log** — completed tests, results, bugs found, fixes applied, session outcomes |
| `.GCC/branches/test_afaire.md` | **Test backlog** — all pending tests to run (plugins, security, core, WhatsApp scenarios) |

### Protocol: after a test session
1. TOOL INVOCATION: Move completed items from `test_afaire.md` to `test.md` with result notes.
2. TOOL INVOCATION: Append any new bugs, regressions, or blocked items to `test.md`.
3. If a bug was fixed or an architecture decision was made → trigger the `decision_milestone`
