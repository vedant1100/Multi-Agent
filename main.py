"""
Multi-Agent Parallel Pipeline — Functional Specialists
Google ADK + Anthropic Claude Models

Architecture:
  SequentialAgent  (ResearchAndSynthesisPipeline)
  ├── ParallelAgent  (ParallelSpecialistAgents)
  │   ├── ResearchAgent   → claude-sonnet-4-6  web search & summarisation
  │   ├── PricingAgent    → claude-sonnet-4-6  competitor pricing analysis
  │   ├── DataAgent       → claude-sonnet-4-6  metrics, trends & statistics
  │   └── CodingAgent     → claude-sonnet-4-6    code generation & execution
  └── SynthesisAgent      → claude-opus-4-6    merges all outputs → final report

Parallel agents write results into shared session state via output_key.
The SynthesisAgent reads all four keys and produces the final deliverable.

Requirements:
    pip install google-adk litellm anthropic

Environment variables:
    ANTHROPIC_API_KEY=<your-key>
    GOOGLE_API_KEY=<your-key>   # Only needed if using google_search tool
"""

import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from gemini_api_key import gemini_agent_2
from gemini_api_key import gemini_agent_3_pro

# Optional: uncomment if GOOGLE_API_KEY is configured
# from google.adk.tools import google_search, built_in_code_execution

# ─────────────────────────────────────────────
# 1. Model Definitions
#    LiteLlm routes to Anthropic via ANTHROPIC_API_KEY.
# ─────────────────────────────────────────────

# Claude Sonnet — fast and cost-effective for focused parallel tasks
GEMINI_LITE = LiteLlm(model=gemini_agent_2)

# Claude Opus — strongest reasoning; used for coding and final synthesis
GEMINI_PRO = LiteLlm(model=gemini_agent_3_pro)


# ─────────────────────────────────────────────
# 2. Parallel Specialist Sub-Agents
#    All four run concurrently inside the ParallelAgent.
#    Each writes its result into session state via output_key,
#    which the SynthesisAgent reads as {key} template variables.
# ─────────────────────────────────────────────

# ── 2a. Research Agent ────────────────────────
# Broad information gathering: market landscape, trends, news.
research_agent = LlmAgent(
    name="ResearchAgent",
    model=GEMINI_LITE,
    instruction="""
    You are a senior market research analyst.

    Task: Gather a comprehensive overview of the topic provided by the user.
    Cover: market landscape, key players, recent news, and emerging trends.

    Instructions:
    - Use the Google Search tool to retrieve current information.
    - Be factual, structured, and concise.
    - Output ONLY a clear summary (3–5 bullet points or 2–3 short paragraphs).
    - Do not include preamble, sign-offs, or meta-commentary.
    """,
    description=(
        "Performs broad market research: landscape, key players, "
        "news, and trend analysis."
    ),
    # tools=[google_search],      # Uncomment when GOOGLE_API_KEY is set
    output_key="research_result",
)

# ── 2b. Pricing Agent ─────────────────────────
# Competitive pricing intelligence: tiers, positioning, benchmarks.
pricing_agent = LlmAgent(
    name="PricingAgent",
    model=GEMINI_LITE,
    instruction="""
    You are a competitive pricing intelligence specialist.

    Task: Analyse the pricing landscape for the topic/product provided by the user.
    Cover: competitor pricing tiers, free-vs-paid models, price anchors,
    discounting patterns, and any publicly available pricing benchmarks.

    Instructions:
    - Use the Google Search tool to find current pricing pages and reports.
    - Structure your output as a concise comparison (table or bullet list).
    - Highlight pricing gaps or opportunities where relevant.
    - Output ONLY the pricing analysis. No preamble or sign-off.
    """,
    description=(
        "Analyses competitor pricing tiers, models, and benchmarks "
        "to surface positioning opportunities."
    ),
    # tools=[google_search],
    output_key="pricing_result",
)

# ── 2c. Data Agent ────────────────────────────
# Quantitative analysis: metrics, statistics, KPIs, growth figures.
data_agent = LlmAgent(
    name="DataAgent",
    model=GEMINI_LITE,
    instruction="""
    You are a quantitative data analyst.

    Task: Find and summarise the key metrics, statistics, and KPIs relevant
    to the topic provided by the user.
    Cover: market size, growth rates, adoption figures, survey data,
    and any authoritative numerical benchmarks.

    Instructions:
    - Use the Google Search tool to locate reports, studies, and data sources.
    - Present data clearly with source attribution where possible.
    - Flag any data quality issues or conflicting figures.
    - Output ONLY a structured data summary (bullets or short paragraphs).
    - Do not include preamble or sign-off.
    """,
    description=(
        "Collects and summarises quantitative metrics, KPIs, market sizes, "
        "and growth statistics."
    ),
    # tools=[google_search],
    output_key="data_result",
)

# ── 2d. Coding Agent ──────────────────────────
# Code generation: scripts, data processing, visualisation stubs.
# Uses Claude Opus for stronger code reasoning.
coding_agent = LlmAgent(
    name="CodingAgent",
    model=GEMINI_LITE,          # Opus for superior code generation quality
    instruction="""
    You are an expert software engineer and data scientist.

    Task: Based on the topic provided by the user, produce ready-to-run Python code
    that does ONE of the following (choose the most useful for the topic):
      (a) Fetches or processes relevant data from a public API or CSV.
      (b) Performs a quantitative analysis or statistical summary.
      (c) Generates a visualisation (matplotlib / plotly) of key metrics.
      (d) Implements a utility function or automation script relevant to the topic.

    Instructions:
    - Write clean, well-commented, production-quality Python code.
    - Include all necessary imports at the top.
    - Add a brief docstring explaining what the script does and how to run it.
    - If using external libraries, note the pip install command in a comment.
    - Output ONLY the code block (no prose before or after).
    """,
    description=(
        "Generates production-quality Python code: data pipelines, "
        "analysis scripts, or visualisations relevant to the task."
    ),
    # tools=[built_in_code_execution],   # Uncomment to execute code in sandbox
    output_key="coding_result",
)


# ─────────────────────────────────────────────
# 3. ParallelAgent — runs all four specialists concurrently
#    Completes once every sub-agent has written its output_key.
# ─────────────────────────────────────────────

parallel_specialist_agents = ParallelAgent(
    name="ParallelSpecialistAgents",
    sub_agents=[
        research_agent,
        pricing_agent,
        data_agent,
        coding_agent,
    ],
    description=(
        "Orchestrates four specialist agents in parallel: "
        "Research, Pricing, Data, and Coding."
    ),
)


# ─────────────────────────────────────────────
# 4. Synthesis Agent — runs AFTER the parallel block
#    Uses Claude Opus for deeper cross-domain reasoning.
#    Reads all four output_keys from session state.
# ─────────────────────────────────────────────

merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_PRO,
    instruction="""
    You are a principal consultant responsible for producing a complete,
    client-ready deliverable from four specialist inputs.

    Your task is to synthesise the outputs below into one cohesive report.
    You must:
    - Be grounded EXCLUSIVELY on the inputs provided — no external knowledge.
    - Cross-reference findings across sections where relevant.
    - Write clearly, professionally, and in plain language.
    - Follow the output format exactly.

    ── SPECIALIST INPUTS ────────────────────────────────────────────────────────

    [Research findings — from ResearchAgent]
    {research_result}

    [Pricing analysis — from PricingAgent]
    {pricing_result}

    [Data & metrics — from DataAgent]
    {data_result}

    [Code artefact — from CodingAgent]
    {coding_result}

    ── OUTPUT FORMAT ─────────────────────────────────────────────────────────────

    ## Integrated Analysis Report

    ### 1. Market Research Summary
    (Source: ResearchAgent)
    [Synthesise the research findings. Highlight the most important insights.]

    ### 2. Pricing Intelligence
    (Source: PricingAgent)
    [Summarise the competitive pricing landscape. Call out key opportunities.]

    ### 3. Data & Metrics
    (Source: DataAgent)
    [Present the key quantitative findings. Note any data quality caveats.]

    ### 4. Code Artefact
    (Source: CodingAgent)
    [Include the generated code exactly as provided, inside a Python code block.]

    ### 5. Strategic Recommendations
    [3–5 actionable recommendations that draw on findings from ALL four sections.
     Each recommendation should reference its supporting evidence.]

    Output ONLY the structured report. No preamble, no sign-off.
    """,
    description=(
        "Synthesises outputs from Research, Pricing, Data, and Coding agents "
        "into a single client-ready integrated analysis report."
    ),
    # No output_key — direct response is the final pipeline output.
)


# ─────────────────────────────────────────────
# 5. SequentialAgent — root entry point
#    Step 1: ParallelSpecialistAgents  → fills session state
#    Step 2: SynthesisAgent            → reads state, writes final report
# ─────────────────────────────────────────────

sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[
        parallel_specialist_agents,   # ← parallel block
        merger_agent,                 # ← synthesis block
    ],
    description=(
        "End-to-end pipeline: runs Research, Pricing, Data, and Coding agents "
        "in parallel (Claude Sonnet/Opus), then synthesises with Claude Opus."
    ),
)

# ADK entry point
root_agent = sequential_pipeline_agent


# ─────────────────────────────────────────────
# Quick local test
# Run: python multi_agent_claude_pipeline.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    # ── Change this prompt to drive the whole pipeline ──
    PIPELINE_TOPIC = (
        "Analyse the current AI coding assistant market "
        "(e.g. GitHub Copilot, Cursor, Tabnine, Amazon CodeWhisperer)."
    )

    async def run_pipeline():
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="specialist_pipeline",
            user_id="user_01",
        )

        runner = Runner(
            agent=root_agent,
            app_name="specialist_pipeline",
            session_service=session_service,
        )

        user_message = Content(
            role="user",
            parts=[Part(text=PIPELINE_TOPIC)],
        )

        print("▶ Starting parallel specialist pipeline...\n")
        print(f"  Topic : {PIPELINE_TOPIC}\n")
        print(f"  Agents: ResearchAgent (Sonnet) | PricingAgent (Sonnet) | "
              f"DataAgent (Sonnet) | CodingAgent (Opus)\n")
        print("─" * 60)

        async for event in runner.run_async(
            user_id="user_01",
            session_id=session.id,
            new_message=user_message,
        ):
            if event.is_final_response():
                print("\n" + "═" * 60)
                print(event.content.parts[0].text)
                print("═" * 60)

    asyncio.run(run_pipeline())