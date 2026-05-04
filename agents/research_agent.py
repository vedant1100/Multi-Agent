from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from gemini_api_key import gemini_agent_2

INSTRUCTION = """
You are a senior market research analyst.
 
Task: Research the market landscape for the topic given by the user.
 
Cover ALL of the following:
- Key players and their market positioning
- Recent news, product launches, or partnerships (last 6–12 months)
- Emerging trends shaping the space
- Any notable challenges or risks in the market
 
Instructions:
- Use the Google Search tool to retrieve current, factual information.
- Structure your output clearly with short paragraphs or bullets per area.
- Be specific — name companies, products, dates where available.
- Output ONLY the research summary. No preamble, no sign-off.
"""

research_agent = LlmAgent(
    name="ResearchAgent",
    model=LiteLlm(model=gemini_agent_2),
    instruction=INSTRUCTION,
    output_key="research_result",
    description=(
        "Performs broad market research: landscape, key players, "
        "recent news, and trend analysis for a given topic."
    )
)