from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
from datetime import datetime
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from pipeline.execution import root_agent

RESULTS_DIR = "results"

async def run(topic: str):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="research_pipeline", user_id="user_01"
    )
    runner = Runner(
        agent=root_agent,
        app_name="research_pipeline",
        session_service=session_service,
    )
    user_message = Content(
        role="user", parts=[Part(text=topic)]
    )
    async for event in runner.run_async(
        user_id="user_01",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response():
            result = event.content.parts[0].text
            os.makedirs(RESULTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(RESULTS_DIR, f"report_{timestamp}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {topic}\n\n")
                f.write(result)
            print(f"Report saved to {filename}")

if __name__ == "__main__":
    asyncio.run(run("Analyse the AI coding assistant market"))