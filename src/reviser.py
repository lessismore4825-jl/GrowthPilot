from src.llm_client import generate_text


def revise_content(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict,
    model_key: str | None = None,
) -> str:
    """
    Revise marketing content based on evaluator feedback.

    The revised content must remain strictly grounded
    in the provided brand information and campaign brief.
    """

    issues = evaluation.get(
        "issues",
        [],
    )

    suggestions = evaluation.get(
        "suggestions",
        [],
    )

    issues_text = "\n".join(
        f"- {issue}"
        for issue in issues
    )

    suggestions_text = "\n".join(
        f"- {suggestion}"
        for suggestion in suggestions
    )


    prompt = f"""
You are a strict AI marketing content editor.

Your task is to revise the original marketing content
according to the evaluation feedback.

The revised content must improve quality WITHOUT
introducing any new unsupported information.


BRAND INFORMATION:

{brand_info}


CAMPAIGN BRIEF:

{campaign_brief}


ORIGINAL CONTENT:

{original_content}


IDENTIFIED ISSUES:

{issues_text}


IMPROVEMENT SUGGESTIONS:

{suggestions_text}


REVISION REQUIREMENTS:

1. Fix all identified factual inconsistencies.

2. Remove invented, exaggerated, or unsupported claims.

3. Do not introduce ANY new factual claim unless
   it is explicitly supported by the BRAND INFORMATION.

4. Do not invent personal experience.

   Do not write statements such as:
   - "I tried it"
   - "After using it..."
   - "My skin improved..."
   - "I have been using it..."
   - "Personally tested..."
   - "I used to have..."

   unless such experience is explicitly provided.

5. Do not invent sensory or performance claims such as:
   - non-sticky
   - gentle
   - non-irritating
   - suitable for all skin types
   - fast absorbing
   - refreshing
   - long-lasting

   unless explicitly provided.

6. Do not invent:
   - ingredients
   - certifications
   - clinical evidence
   - research results
   - prices
   - technical specifications
   - safety claims
   - medical claims
   - health claims
   - user outcomes

7. Correct altered numbers and specifications.

8. Include important provided selling points
   that are missing.

9. Improve brand alignment and tone.

10. Match the requested platform and audience
    through writing style only.

    Platform style does NOT justify inventing
    personal experience or product facts.

11. Preserve useful parts of the original content
    when they are already correct.

12. Brand restrictions always override
    conflicting campaign requests.

13. Follow the requested content length
    as closely as reasonably possible.

14. If a desirable marketing claim is not supported
    by the brand information, leave it out.

15. Do not explain your revision process.

16. Return ONLY the revised marketing content.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.1,
    )