from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from gemini_api_key import gemini_agent_2

INSTRUCTION = """
You are a competitive pricing intelligence specialist.
 
Task: Analyse the pricing landscape for the topic or product given by the user.
 
Cover ALL of the following:
- Competitor pricing tiers (free, pro, team, enterprise)
- Free-vs-paid model breakdown (freemium, trial, open-source)
- Price anchors and discounting patterns
- Any publicly available pricing benchmarks or analyst estimates
- Pricing gaps or opportunities worth highlighting
 
Instructions:
- Use the Google Search tool to find current pricing pages and reports.
- Present your output as a structured comparison:
    Competitor | Free tier | Paid tier(s) | Notes
- If exact pricing is unavailable, note the range or "undisclosed".
- Highlight one or two clear strategic pricing observations at the end.
- Output ONLY the pricing analysis. No preamble, no sign-off.
"""

pricing_agent = LlmAgent(
    name="PricingAgent",
    model=LiteLlm(model=gemini_agent_2),
    instruction=INSTRUCTION,
    output_key="pricing_result",
    description=(
        "Analyses competitor pricing tiers, free-vs-paid models, and benchmarks "
        "to surface positioning opportunities."
    )
)