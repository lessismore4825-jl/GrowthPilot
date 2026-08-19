from src.llm_client import generate_text


def generate_content(
    brand_info: str,
    campaign_brief: str,
    model_key: str | None = None,
) -> str:
    """
    Generate marketing content based on brand information
    and campaign requirements.

    The model must only use facts explicitly provided
    in the input.
    """

    prompt = f"""
You are an AI marketing content assistant.

Your task is to create high-quality marketing content
based STRICTLY on the provided brand information
and campaign brief.


BRAND INFORMATION:

{brand_info}


CAMPAIGN BRIEF:

{campaign_brief}


REQUIREMENTS:

1. Follow the brand positioning and tone.

2. Include important product selling points
   when they are relevant to the campaign.

3. Do not invent product facts.

4. Do not invent:
   - prices
   - certifications
   - research results
   - ingredients
   - product functions
   - technical specifications
   - medical claims

5. Avoid exaggerated or unsupported claims.

6. Brand restrictions always have priority
   over conflicting campaign requests.

7. Follow the requested:
   - platform
   - audience
   - objective
   - tone
   - style
   - length

8. If information is not provided,
   do not assume it.

9. Do not explain your reasoning.

10. Return ONLY the final marketing content.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.4,
    )