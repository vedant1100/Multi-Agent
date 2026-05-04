from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from agents.research_agent  import research_agent
from agents.pricing_agent   import pricing_agent
from agents.data_agent      import data_agent
from agents.coding_agent    import coding_agent
from agents.merger_agent import synthesis_agent
 
# ── Parallel block: all 4 specialists run concurrently ─────────────────────────
parallel_block = ParallelAgent(
    name="ParallelSpecialistAgents",
    sub_agents=[
        research_agent,
        pricing_agent,
        data_agent,
        coding_agent,
    ],
    description=(
        "Runs Research, Pricing, Data, and Coding agents in parallel. "
        "Completes when all four have written their output_key to session state."
    ),
)
 
# ── Root agent: sequential → parallel first, then synthesis ───────────────────
root_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[
        parallel_block,     # step 1: fills session state concurrently
        synthesis_agent,    # step 2: reads state, writes final report
    ],
    description=(
        "End-to-end pipeline: runs four Claude specialists in parallel, "
        "then synthesises with Claude Opus into an integrated report."
    ),
)