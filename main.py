import asyncio
import os
import sys
from datetime import datetime
from typing import List, Optional

import anthropic
import logfire
import uvicorn
from dbos import DBOS, DBOSConfig, WorkflowHandleAsync
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable Logfire instrumentation if LOGFIRE_TOKEN is set
enable_logfire = False
if os.environ.get("LOGFIRE_TOKEN"):
    print("Enabling Logfire instrumentation")
    enable_logfire = True
    logfire.configure(service_name="research-agent")

# Validate required environment variables
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY environment variable not set")
    sys.exit(1)

anthropic_client = anthropic.AsyncAnthropic()
google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

AGENT_STATUS = "agent_status"
APPROVAL_TOPIC = "approval"


# ── Models ───────────────────────────────────────────────────────────────────

class AgentStartRequest(BaseModel):
    query: str


class ApprovalRequest(BaseModel):
    action: str  # "finish" or "research_more"
    prompt: Optional[str] = None


class SearchStepStatus(BaseModel):
    """Kept for backward compatibility with pickled events stored in SQLite."""
    search_terms: str
    completed: bool = False


class AgentStatus(BaseModel):
    created_at: str
    query: str
    report: Optional[str]
    status: str
    agent_id: str = ""


class SearchInfo(BaseModel):
    workflow_id: str
    search_terms: str
    status: str
    completed: bool


# ── LLM steps ─────────────────────────────────────────────────────────────────

@DBOS.step(retries_allowed=True, max_attempts=20)
async def plan_step(query: str) -> dict:
    """Call Claude with tool use to produce a structured research plan."""
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system="Analyze the user's query and design a plan for deep research to answer their query.",
        tools=[{
            "name": "research_plan",
            "description": "Output a structured research plan",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the research plan"
                    },
                    "web_search_steps": {
                        "type": "array",
                        "maxItems": 3,
                        "description": "List of web searches to perform",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search_terms": {"type": "string"}
                            },
                            "required": ["search_terms"]
                        }
                    },
                    "analysis_instructions": {
                        "type": "string",
                        "description": "Instructions for synthesizing the search results into a final report"
                    }
                },
                "required": ["summary", "web_search_steps", "analysis_instructions"]
            }
        }],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": query}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "research_plan":
            return block.input
    raise RuntimeError("plan_step: Claude did not return a research_plan tool call")


@DBOS.step(retries_allowed=True, max_attempts=20)
async def search_step(search_terms: str) -> str:
    """Call Gemini with native Google Search grounding."""
    response = await google_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=search_terms,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Perform a web search for the given terms and return a report on the results.\n\n"
                "Structure your response as follows:\n"
                "1. Start with the most important findings and key facts immediately — put the highest-value information first.\n"
                "2. Follow with supporting details and context.\n"
                "3. End with a \"## Sources\" section listing every URL visited, "
                "formatted as markdown links: [source title](URL).\n\n"
                "Be concise. Avoid filler and repetition."
            ),
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return response.text


_PUBLISH_REPORT_TOOL = {
    "name": "publish_report",
    "description": "Publish the final research report when you have sufficient information to answer the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "report": {"type": "string", "description": "Complete research report in markdown, starting with an executive summary."}
        },
        "required": ["report"]
    }
}

_REQUEST_MORE_SEARCHES_TOOL = {
    "name": "request_more_searches",
    "description": "Request additional web searches when the current results are insufficient.",
    "input_schema": {
        "type": "object",
        "properties": {
            "search_terms": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string"},
                "description": "Specific search queries needed to fill gaps in the research."
            }
        },
        "required": ["search_terms"]
    }
}

_ANALYZE_SYSTEM = (
    "Analyze the provided research and either publish a final report or request additional searches.\n\n"
    "Call publish_report when you have enough information to give a thorough, well-sourced answer.\n"
    "Call request_more_searches when there are clear gaps — be specific about what you still need.\n\n"
    "Your report should start with an executive summary, then a concise analysis of the findings. "
    "Include links to original sources whenever possible."
)


@DBOS.step(retries_allowed=True, max_attempts=20)
async def analyze_step(query: str, search_results: List[str], instructions: str, force_publish: bool = False) -> dict:
    """Call Claude to either publish a report or request more searches.

    Returns {"action": "publish", "report": "..."} or {"action": "search_more", "search_terms": [...]}.
    """
    parts = [f"<query>{query}</query>\n"]
    for i, result in enumerate(search_results, 1):
        parts.append(f"<search_result_{i}>\n{result}\n</search_result_{i}>\n")
    parts.append(f"<analysis_instructions>{instructions}</analysis_instructions>")

    tools = [_PUBLISH_REPORT_TOOL] if force_publish else [_PUBLISH_REPORT_TOOL, _REQUEST_MORE_SEARCHES_TOOL]
    tool_choice: dict = {"type": "tool", "name": "publish_report"} if force_publish else {"type": "any"}

    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8192 * 2,
        system=_ANALYZE_SYSTEM,
        tools=tools,
        tool_choice=tool_choice,
        messages=[{"role": "user", "content": "\n".join(parts)}],
    )
    for block in response.content:
        if block.type == "tool_use":
            if block.name == "publish_report":
                return {"action": "publish", "report": block.input["report"]}
            if block.name == "request_more_searches":
                return {"action": "search_more", "search_terms": block.input["search_terms"]}
    raise RuntimeError("analyze_step: Claude did not call publish_report or request_more_searches")


# ── DBOS workflows ────────────────────────────────────────────────────────────

MAX_EXTRA_SEARCH_ROUNDS = 2  # max rounds of additional searches the analyzer can request
MAX_SEARCH_RESULT_CHARS = 4000  # truncation limit per search result fed to the analyzer


@DBOS.workflow()
async def search_workflow(search_terms: str) -> str:
    return await search_step(search_terms)


@DBOS.workflow()
async def analyze_workflow(query: str, search_results: List[str], instructions: str) -> str:
    """Iteratively analyzes results, fanning out extra searches as needed, until ready to publish."""
    accumulated = list(search_results)

    for round_num in range(MAX_EXTRA_SEARCH_ROUNDS + 1):
        force = round_num >= MAX_EXTRA_SEARCH_ROUNDS
        trimmed = [r[:MAX_SEARCH_RESULT_CHARS] for r in accumulated]
        result = await analyze_step(query, trimmed, instructions, force_publish=force)

        if result["action"] == "publish":
            return result["report"]

        # Fan out additional searches in parallel
        handles: List[WorkflowHandleAsync[str]] = []
        for terms in result["search_terms"]:
            handle = await DBOS.start_workflow_async(search_workflow, terms)
            handles.append(handle)
        extra = [await h.get_result() for h in handles]
        accumulated.extend(extra)

    # Should not be reached (force_publish=True on last round guarantees a report)
    raise RuntimeError("analyze_workflow: exhausted search rounds without publishing")


@DBOS.workflow()
async def deep_research(query: str) -> str:
    created_at = datetime.now().isoformat()
    current_query = query
    display_query = query
    current_report: Optional[str] = None

    while True:
        agent_status = AgentStatus(
            created_at=created_at,
            query=display_query,
            report=current_report,
            status="PLANNING",
        )
        await DBOS.set_event_async(AGENT_STATUS, agent_status)
        plan = await plan_step(current_query)

        agent_status.status = "SEARCHING"
        await DBOS.set_event_async(AGENT_STATUS, agent_status)
        task_handles: List[WorkflowHandleAsync[str]] = []
        for step in plan["web_search_steps"]:
            handle = await DBOS.start_workflow_async(search_workflow, step["search_terms"])
            task_handles.append(handle)

        search_results = [await h.get_result() for h in task_handles]

        agent_status.status = "ANALYZING"
        await DBOS.set_event_async(AGENT_STATUS, agent_status)
        analyze_handle = await DBOS.start_workflow_async(
            analyze_workflow, current_query, search_results, plan["analysis_instructions"]
        )
        current_report = await analyze_handle.get_result()

        agent_status.report = current_report
        agent_status.status = "PENDING_APPROVAL"
        await DBOS.set_event_async(AGENT_STATUS, agent_status)

        # Wait for the user to finish or request more research (up to 30 days)
        approval = await DBOS.recv_async(APPROVAL_TOPIC, timeout_seconds=30 * 24 * 3600)

        if approval is None or approval.get("action") == "finish":
            agent_status.status = "COMPLETED"
            await DBOS.set_event_async(AGENT_STATUS, agent_status)
            return current_report

        # Research more: build a context-aware query for the next iteration
        additional_prompt = approval.get("prompt") or ""
        display_query = additional_prompt
        current_query = (
            f"<previous_research_summary>\n{current_report}\n</previous_research_summary>\n\n"
            f"Additional research requested: {additional_prompt}"
        )


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

@app.post("/agents")
async def start_agent(request: AgentStartRequest):
    await DBOS.start_workflow_async(deep_research, request.query)
    return {"ok": True}


@app.post("/agents/{agent_id}/approve")
async def approve_agent(agent_id: str, request: ApprovalRequest):
    await DBOS.send_async(agent_id, {"action": request.action, "prompt": request.prompt}, APPROVAL_TOPIC)
    return {"ok": True}


@app.get("/agents", response_model=list[AgentStatus])
async def list_agents():
    agent_workflows = await DBOS.list_workflows_async(
        name=deep_research.__qualname__,
        sort_desc=True,
    )
    statuses: list[AgentStatus] = await asyncio.gather(
        *[DBOS.get_event_async(w.workflow_id, AGENT_STATUS) for w in agent_workflows]
    )
    for workflow, status in zip(agent_workflows, statuses):
        status.status = workflow.status if workflow.status in ("ERROR", "CANCELLED") else status.status
        status.agent_id = workflow.workflow_id
    return statuses


@app.get("/agents/{agent_id}/searches", response_model=list[SearchInfo])
async def get_agent_searches(agent_id: str):
    def _wf_to_search_info(wf) -> SearchInfo:
        terms = wf.input["args"][0] if wf.input and wf.input.get("args") else ""
        return SearchInfo(
            workflow_id=wf.workflow_id,
            search_terms=terms,
            status=wf.status,
            completed=wf.status == "SUCCESS",
        )

    # Initial plan searches: search_workflow children of deep_research
    plan_wfs = await DBOS.list_workflows_async(
        parent_workflow_id=agent_id,
        name=search_workflow.__qualname__,
        load_input=True,
    )
    searches = [_wf_to_search_info(wf) for wf in plan_wfs]

    # Extra searches: search_workflow children of analyze_workflow children
    analyze_wfs = await DBOS.list_workflows_async(
        parent_workflow_id=agent_id,
        name=analyze_workflow.__qualname__,
    )
    extra_wf_lists = await asyncio.gather(*[
        DBOS.list_workflows_async(
            parent_workflow_id=awf.workflow_id,
            name=search_workflow.__qualname__,
            load_input=True,
        )
        for awf in analyze_wfs
    ])
    for wf_list in extra_wf_lists:
        searches.extend(_wf_to_search_info(wf) for wf in wf_list)

    return searches


@app.get("/workflows/{workflow_id}/output")
async def get_workflow_output(workflow_id: str):
    wfs = await DBOS.list_workflows_async(workflow_ids=[workflow_id], load_output=True)
    if not wfs:
        raise HTTPException(status_code=404, detail="Workflow not found")
    output = wfs[0].output
    return {"output": str(output) if output is not None else ""}


@app.post("/crash")
async def crash_app():
    os._exit(1)


if __name__ == "__main__":
    config: DBOSConfig = {
        "name": "research-agent-prototype",
        "system_database_url": os.environ.get("DBOS_SYSTEM_DATABASE_URL"),
        "conductor_key": os.environ.get("DBOS_CONDUCTOR_KEY"),
        "enable_otlp": enable_logfire,
        "application_version": "0.1.0",
    }
    DBOS(config=config)
    DBOS.launch()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
