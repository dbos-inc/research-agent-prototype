# Research Agent

A durable deep-research agent built with [DBOS](https://dbos.dev), Anthropic Claude, and Google Gemini. Given a research query, it plans a set of web searches, runs them in parallel, and synthesizes the results into a report. This is followed by a human-in-the-loop approval step. The user can either approve the report or request a follow-on search with the current context. All steps are durable — if the app crashes mid-run, it resumes exactly where it left off.

## Architecture

- **Plan** — Claude (Sonnet) designs a structured research plan with up to 5 search queries
- **Search** — Gemini 2.5 Flash runs each search in parallel using native Google Search grounding
- **Analyze** — Claude (Sonnet) synthesizes results into a report, or requests more searches if needed (up to 2 extra rounds)
- **Approve** — the user reviews the report and either finishes or requests a follow-up research iteration

All LLM calls are wrapped as DBOS steps (durable, retriable). The orchestration loop is a DBOS workflow (survives crashes and restarts).

## This app showcases the following features of DBOS
 - **Durability:** you can restart the app at any point and it will resume from where it left off
 - **Parallelism:** executing multiple search sub-workflows and gathering their results reliably
 - **Observability:** workflow state is easily accessible from Conductor and the App UI
 - **Human-in-the-Loop Approval:** workflows waiting for approval persist across restarts.

## Prerequisites

- Python 3.13+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A running PostgreSQL instance (for DBOS system state), **or** omit `DBOS_SYSTEM_DATABASE_URL` to use the local SQLite fallback

## Setup

**1. Clone and enter the repo**

```bash
git clone <repo-url>
cd research-agent-prototype
```

**2. Install Python dependencies**

```bash
uv sync
```

**3. Install frontend dependencies**

```bash
cd frontend && npm install && cd ..
```

**4. Set environment variables**

Create a `.env` file or export in your shell:

```bash
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Optional — omit to use local SQLite
DBOS_SYSTEM_DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Connect to DBOS Conductor: get an API key at console.dbos.dev
DBOS_CONDUCTOR_KEY=...
```

- `ANTHROPIC_API_KEY` — get one at [console.anthropic.com](https://console.anthropic.com)
- `GOOGLE_API_KEY` — get one at [aistudio.google.com](https://aistudio.google.com) (enable the Generative Language API)

## Running

```bash
./launch_app.sh
```

This starts both the backend (port 8000) and the frontend dev server (port 5173). Open [http://localhost:5173](http://localhost:5173).

To start them separately:

```bash
# Backend
uv run python main.py

# Frontend (in another terminal)
cd frontend && npm run dev
```

## Usage

1. Enter a research query and click **Launch**
2. The agent plans, searches, and analyzes. It starts and displays search sub-workflows. Click on each search to see its results.
3. When the report is ready, either:
   - **Finish** — mark the research complete
   - **Research More** — provide an additional prompt to kick off another iteration, building on the existing report

The **💥 Crash App** button kills the backend immediately — useful for testing DBOS crash recovery. Restart the backend and the in-progress workflow resumes from where it left off.
