import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def generate_content(brand_info: str, campaign_brief: str) -> str:
    prompt = f"""
You are an AI marketing content assistant.

Your task is to create high-quality marketing content based strictly on the
brand information and campaign brief provided below.

BRAND INFORMATION:
{brand_info}

CAMPAIGN BRIEF:
{campaign_brief}

Requirements:
1. Follow the brand positioning and tone.
2. Include the important product selling points.
3. Do not invent product facts that are not provided.
4. Avoid exaggerated or unsupported claims.
5. Produce polished marketing content suitable for the requested platform.

Return only the final marketing content.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text