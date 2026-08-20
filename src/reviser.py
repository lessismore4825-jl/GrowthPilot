from src.llm_client import generate_text


# =========================================================
# Helpers
# =========================================================

def format_compliance_findings(
    findings: list,
) -> str:
    """Render structured blocking compliance findings for edit prompts."""

    if not findings:
        return "No blocking compliance findings."

    sections = []

    for index, finding in enumerate(
        findings,
        start=1,
    ):
        sections.append(
            f"""
FINDING {index}

Problematic wording:
{finding.get("evidence", "")}

Policy source:
{finding.get("policy_source", "")}

Policy basis:
{finding.get("policy_basis", "")}

Reason:
{finding.get("reason", "")}

Required action:
{finding.get("required_action", "")}
""".strip()
        )

    return "\n\n".join(sections)


def format_requirement_findings(
    findings: list,
) -> str:
    """Render structured mandatory campaign requirement findings."""

    if not findings:
        return "No mandatory requirement findings."

    sections = []

    for index, finding in enumerate(
        findings,
        start=1,
    ):
        sections.append(
            f"""
REQUIREMENT {index}

Requirement ID:
{finding.get("requirement_id", "")}

Missing mandatory content:
{finding.get("requirement", "")}

Match mode:
{finding.get("match_mode", "")}

Reason:
{finding.get("reason", "")}

Required action:
{finding.get("required_action", "")}
""".strip()
        )

    return "\n\n".join(sections)


def format_advisory_findings(
    findings: list,
) -> str:
    """Render non-blocking advisory findings and their provenance."""

    if not findings:
        return "No advisory findings."

    sections = []

    for index, finding in enumerate(
        findings,
        start=1,
    ):
        sections.append(
            f"""
SUGGESTION {index}

Area:
{finding.get("area", "General")}

Relevant wording:
{finding.get("evidence", "")}

Reason:
{finding.get("reason", "")}

Suggestion:
{finding.get("suggestion", "")}

Basis type:
{finding.get("basis_type", "GENERAL_HEURISTIC")}

Basis source:
{finding.get("basis_source", "")}

Basis quote:
{finding.get("basis_quote", "")}
""".strip()
        )

    return "\n\n".join(sections)


def _policy_context_display(
    policy_context: str,
) -> str:
    """Return stable display text for optional policy context."""

    value = (
        policy_context
        or ""
    ).strip()

    if value:
        return value

    return "No additional external policy was provided."


def _content_origin_instruction(
    evaluation: dict,
) -> str:
    """Return revision rules appropriate to the submitted content origin."""

    content_origin = str(
        evaluation.get(
            "content_origin",
            "generated",
        )
        or "generated"
    ).strip().lower()

    if content_origin == "creator_draft":
        return """
The original content is CREATOR-SUBMITTED CONTENT.

Preserve existing first-person creator voice and personal-experience wording
when it is not itself one of the confirmed mandatory problems.

Do NOT assume a creator's existing first-person statement is fabricated merely
because Brand Information does not independently prove that personal experience.

However, do not create NEW creator traits, experiences, endorsements or outcomes.
""".strip()

    return """
The original content may be AI-generated.

Do not invent new first-person experiences, endorsements, user outcomes or
personal traits during revision.
""".strip()


# =========================================================
# Minimal Compliance Fix
# =========================================================

def fix_compliance_issues(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict,
    policy_context: str = "",
    model_key: str | None = None,
) -> str:
    """
    Perform a Minimal Compliance Edit.

    Only confirmed blocking compliance findings are mandatory here.
    Requirement completion and optional quality optimization remain separate.
    """

    findings = (
        evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    if not findings:
        return original_content

    findings_text = format_compliance_findings(
        findings
    )

    policy_display = _policy_context_display(
        policy_context
    )

    origin_instruction = _content_origin_instruction(
        evaluation
    )

    prompt = f"""
You are a strict and conservative marketing compliance editor.

Your task is ONLY to perform a:

MINIMAL COMPLIANCE EDIT

Fix every confirmed Blocking Compliance Finding using the smallest necessary
textual change.

Do NOT treat this as a general rewrite.


============================================================
CONTENT ORIGIN
============================================================

{origin_instruction}


============================================================
BRAND INFORMATION
============================================================

{brand_info}


============================================================
CAMPAIGN BRIEF
============================================================

{campaign_brief}


============================================================
ADDITIONAL POLICY CONTEXT
============================================================

{policy_display}


============================================================
ORIGINAL CONTENT
============================================================

{original_content}


============================================================
BLOCKING COMPLIANCE FINDINGS
============================================================

{findings_text}


============================================================
PRIMARY OBJECTIVE
============================================================

Correct ONLY the confirmed blocking compliance problems.

Do NOT use this task for:

- creative rewriting
- style optimization
- platform optimization
- selling-point enrichment
- engagement optimization
- copy expansion
- optional advisory fixes
- unrelated campaign requirement completion


============================================================
SOURCE-OF-TRUTH RULE
============================================================

The confirmed Evidence and Policy Basis are the source of truth.

A Judge's free-form REQUIRED_ACTION is guidance only.

If any part of REQUIRED_ACTION would require an unsupported new claim,
new creator experience, new product benefit or stronger promise:

IGNORE that unsafe part.

Use the smallest correction supported by the supplied sources instead.


============================================================
MINIMAL EDIT RULES
============================================================

1. MINIMIZE CHANGES

Change or remove only the wording necessary to resolve each confirmed
blocking conflict.

Do not rewrite an entire sentence when a smaller edit is sufficient.

Do not rewrite the whole piece unless the confirmed violations are so extensive
that a smaller edit cannot produce coherent content.


2. PRESERVE CORRECT CONTENT

Preserve whenever possible:

- correct product facts
- correct campaign content
- sentence structure
- creator voice
- first-person perspective already present
- platform style
- tone
- correct selling points
- unaffected wording


3. DO NOT ADD NEW MARKETING CLAIMS

Do NOT add new:

- product benefits
- product functions
- performance claims
- sensory claims
- emotional outcomes
- user outcomes
- lifestyle outcomes
- technical properties
- safety claims
- health claims
- medical claims
- superiority claims
- market-leadership claims


4. NO UNSUPPORTED INFERENCE

Do not infer one property from another.

Examples:

"lightweight" does not automatically mean "fast absorbing".

"non-greasy" does not automatically mean "non-sticky".

"10,000mAh" does not automatically mean "battery lasts all day".

"up to 24 hours hydration" does not mean "guaranteed hydration all day".


5. CREATOR EXPERIENCE SAFETY

Preserve existing creator first-person wording when it is not the confirmed
problem.

Do NOT invent NEW wording such as:

- "I tried it..."
- "After using it..."
- "My skin became..."
- "It solved my..."

unless that experience already exists in the submitted content or is explicitly
supplied as approved source material.


6. NO INVENTED EVIDENCE

Never invent:

- clinical studies
- research results
- certifications
- expert endorsements
- awards
- statistics
- customer reviews
- sales rankings
- prices
- ingredients
- specifications
- testing results


7. PRESERVE FACTUAL QUALIFIERS

Preserve qualifiers such as:

- up to
- approximately
- may
- designed for

Do not strengthen them.

Example:

"up to 24 hours" must not become "24 hours guaranteed".


8. DO NOT STRENGTHEN CLAIMS

Prefer deletion or direct correction using an explicitly supplied fact.

Do not replace a false claim with a stronger promotional claim.


9. BRAND AND POLICY PRIORITY

If Campaign Brief conflicts with Brand Information or supplied policy context,
the safer explicit rule takes priority.


10. DO NOT FIX ADVISORIES HERE

Do not proactively fix optional tone, platform, creativity, wording or engagement
Advisories.


11. DO NOT COMPLETE UNRELATED REQUIREMENTS HERE

Requirement Findings have a separate mandatory action.

Only make a requirement-related change here if it is strictly necessary to keep
the compliance correction coherent.


============================================================
WHEN DELETION IS ENOUGH
============================================================

If deleting the problematic phrase produces coherent usable content:

DELETE IT.

Do not replace it with a new promotional claim merely to preserve length.


============================================================
WHEN REPLACEMENT IS NECESSARY
============================================================

If a false statement can be corrected directly using an explicit supplied fact,
replace it with that fact and nothing stronger.

Example:

Original:
"72 hours of hydration"

Supplied fact:
"up to 24 hours of hydration"

Preferred correction:
"up to 24 hours of hydration"


============================================================
FINAL SELF-CHECK
============================================================

Before returning, silently verify:

1. Every confirmed Blocking Finding has been addressed.
2. No unsupported factual claim was added.
3. No new creator/user experience was invented.
4. No unsupported property was inferred.
5. No factual qualifier was strengthened.
6. Correct original content was preserved wherever possible.
7. Optional Advisory issues were not opportunistically rewritten.
8. Changes were minimal.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the revised marketing content.

Do NOT explain changes.
Do NOT provide reasoning.
Do NOT provide edit notes.
Do NOT use Markdown fences.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.0,
    )


# =========================================================
# Minimal Requirement Completion
# =========================================================

def complete_requirements(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict,
    policy_context: str = "",
    model_key: str | None = None,
) -> str:
    """
    Perform Minimal Requirement Completion.

    Missing mandatory Campaign Requirements are mandatory product actions,
    but they are not labeled as Compliance violations.
    """

    findings = (
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    if not findings:
        return original_content

    findings_text = format_requirement_findings(
        findings
    )

    policy_display = _policy_context_display(
        policy_context
    )

    origin_instruction = _content_origin_instruction(
        evaluation
    )

    prompt = f"""
You are a conservative Campaign Content Editor.

Your task is ONLY to perform:

MINIMAL REQUIREMENT COMPLETION

The content is already written.

Add every confirmed missing mandatory Campaign Requirement using the smallest
necessary change.

Do NOT turn this into a general rewrite.


============================================================
CONTENT ORIGIN
============================================================

{origin_instruction}


============================================================
BRAND INFORMATION
============================================================

{brand_info}


============================================================
CAMPAIGN BRIEF
============================================================

{campaign_brief}


============================================================
ADDITIONAL POLICY CONTEXT
============================================================

{policy_display}


============================================================
ORIGINAL CONTENT
============================================================

{original_content}


============================================================
CONFIRMED MISSING MANDATORY REQUIREMENTS
============================================================

{findings_text}


============================================================
PRIMARY OBJECTIVE
============================================================

Complete ONLY the confirmed missing mandatory Campaign Requirements.

This task is NOT:

- a general rewrite
- a compliance rewrite
- creative enrichment
- style optimization
- engagement optimization
- optional advisory optimization
- selling-point expansion beyond the confirmed requirement


============================================================
REQUIREMENT ACTION SAFETY
============================================================

Requirement REQUIRED_ACTION is generated deterministically from the structured
Requirement and may be followed as the mandatory editing instruction.

Do not add any extra benefit, guarantee, outcome or interpretation beyond the
required content itself.

Example:

Requirement:
"BPA-free"

Safe addition:
"BPA-free"

Unsafe addition:
"BPA-free so it is completely safe for all daily use"

Do NOT add the unsupported safety conclusion.


============================================================
MINIMAL REQUIREMENT RULES
============================================================

1. ADD EVERY CONFIRMED REQUIREMENT

Complete each confirmed Requirement Finding.


2. MINIMIZE CHANGES

Prefer the smallest natural insertion into existing content.

Do not rewrite the whole piece unless necessary for coherence.


3. PRESERVE CREATOR VOICE

Preserve whenever possible:

- creator tone
- first-person perspective
- sentence structure
- platform style
- existing wording
- correct product facts
- correct selling points


4. NO GENERAL OPTIMIZATION

Do not proactively improve:

- hook
- creativity
- hashtags
- emojis
- engagement
- overall style
- unrelated length issues

unless strictly necessary to insert the required content coherently.


5. DO NOT INVENT FACTS

Never add unsupported:

- product functions
- product benefits
- ingredients
- specifications
- prices
- certifications
- studies
- statistics
- guarantees
- performance outcomes
- medical outcomes
- health outcomes


6. CREATOR EXPERIENCE SAFETY

Do not invent a new creator:

- personal trait
- skin type
- lifestyle fact
- personal experience
- personal result
- endorsement history

Existing first-person content may be preserved.


7. EXACT REQUIREMENTS

If MATCH_MODE=EXACT:

include the required content exactly.

Typical examples:

- campaign hashtag
- mandatory disclosure
- required slogan
- specified product name


8. SEMANTIC REQUIREMENTS

If MATCH_MODE=SEMANTIC:

express the required concept naturally.

Exact wording is not required.

Do not keyword-stuff when equivalent meaning can be expressed naturally.


9. PRESERVE QUALIFIERS

Never strengthen supplied facts.

Example:

"up to 18 hours" must not become "18 hours guaranteed".


10. DO NOT FIX ADVISORIES

Do not act on optional Advisory Findings during Requirement Completion.


============================================================
FINAL SELF-CHECK
============================================================

Before returning, silently verify:

1. Every confirmed Requirement is now present.
2. No unrelated claim was added.
3. No unsupported benefit was attached to the Requirement.
4. No new creator experience or trait was invented.
5. Existing creator voice was preserved.
6. No optional Advisory optimization occurred.
7. Changes were minimal.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the completed marketing content.

Do NOT explain changes.
Do NOT provide reasoning.
Do NOT provide edit notes.
Do NOT use Markdown fences.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.0,
    )


# =========================================================
# Combined Mandatory Fix
# =========================================================

def fix_mandatory_issues(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict,
    policy_context: str = "",
    model_key: str | None = None,
) -> str:
    """
    Resolve confirmed Compliance + Requirement findings in one conservative pass.

    This avoids two sequential LLM rewrites when both mandatory issue types exist.
    """

    compliance_findings = (
        evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    requirement_findings = (
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    if (
        not compliance_findings
        and not requirement_findings
    ):
        return original_content

    if (
        compliance_findings
        and not requirement_findings
    ):
        return fix_compliance_issues(
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            original_content=original_content,
            evaluation=evaluation,
            policy_context=policy_context,
            model_key=model_key,
        )

    if (
        requirement_findings
        and not compliance_findings
    ):
        return complete_requirements(
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            original_content=original_content,
            evaluation=evaluation,
            policy_context=policy_context,
            model_key=model_key,
        )

    compliance_text = format_compliance_findings(
        compliance_findings
    )

    requirement_text = format_requirement_findings(
        requirement_findings
    )

    policy_display = _policy_context_display(
        policy_context
    )

    origin_instruction = _content_origin_instruction(
        evaluation
    )

    prompt = f"""
You are a strict and conservative Marketing Content Editor.

Perform ONE:

MINIMAL MANDATORY EDIT

The edit must do BOTH:

A. Correct every confirmed Blocking Compliance Finding.
B. Complete every confirmed missing Mandatory Campaign Requirement.

Do NOT use this task for optional quality optimization.


============================================================
CONTENT ORIGIN
============================================================

{origin_instruction}


============================================================
BRAND INFORMATION
============================================================

{brand_info}


============================================================
CAMPAIGN BRIEF
============================================================

{campaign_brief}


============================================================
ADDITIONAL POLICY CONTEXT
============================================================

{policy_display}


============================================================
ORIGINAL CONTENT
============================================================

{original_content}


============================================================
BLOCKING COMPLIANCE FINDINGS
============================================================

{compliance_text}


============================================================
MISSING MANDATORY REQUIREMENTS
============================================================

{requirement_text}


============================================================
MANDATORY EDIT RULES
============================================================

1. Correct every confirmed Blocking Compliance Finding.

2. Complete every confirmed Mandatory Requirement.

3. Make the smallest TOTAL change possible.

4. Preserve correct facts, creator voice, platform style and unaffected wording.

5. Treat Compliance Evidence + Policy Basis as the source of truth.

6. Treat a Compliance REQUIRED_ACTION as guidance only. If it would introduce
   unsupported material, ignore the unsafe part and use the smallest supported
   correction.

7. Treat Requirement REQUIRED_ACTION as a deterministic instruction for adding
   only the missing required content.

8. Do NOT add extra benefits or claims around a Requirement.

9. Do NOT fix optional Advisory Findings.

10. Do NOT introduce unsupported:

- product claims
- functions
- benefits
- creator traits
- creator experiences
- guarantees
- performance outcomes
- health outcomes
- medical outcomes

11. Never invent:

- studies
- certifications
- statistics
- prices
- ingredients
- specifications
- endorsements
- testing results

12. Preserve factual qualifiers exactly.

13. Never strengthen supplied facts.

14. Prefer deletion when deletion safely resolves a false or prohibited claim.

15. If replacing a false claim, use only an explicit supplied fact and nothing
    stronger.

16. Preserve existing first-person creator wording when it is not itself a
    confirmed mandatory problem.


============================================================
FINAL SELF-CHECK
============================================================

Before returning, silently verify:

1. All Compliance Findings are corrected.
2. All Requirement Findings are completed.
3. No unsupported new claim was added.
4. No extra benefit was attached to a Requirement.
5. No new creator experience was invented.
6. Creator voice was preserved wherever possible.
7. No optional Advisory optimization occurred.
8. Changes are minimal.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the revised marketing content.

Do NOT explain changes.
Do NOT provide reasoning.
Do NOT provide edit notes.
Do NOT use Markdown fences.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.0,
    )


# =========================================================
# Optional Quality Optimization
# =========================================================

def optimize_quality(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict,
    policy_context: str = "",
    model_key: str | None = None,
) -> str:
    """
    Optional human-triggered quality optimization.

    Advisory findings remain non-mandatory.
    """

    advisory_findings = (
        evaluation.get(
            "advisory_findings",
            [],
        )
        or []
    )

    if not advisory_findings:
        return original_content

    advisory_text = format_advisory_findings(
        advisory_findings
    )

    policy_display = _policy_context_display(
        policy_context
    )

    origin_instruction = _content_origin_instruction(
        evaluation
    )

    prompt = f"""
You are a careful Marketing Content Editor.

This task is an:

OPTIONAL QUALITY OPTIMIZATION

It is triggered only because a Human user chose to act on Advisory Findings.

Advisory Findings are suggestions, not mandatory violations.


============================================================
CONTENT ORIGIN
============================================================

{origin_instruction}


============================================================
BRAND INFORMATION
============================================================

{brand_info}


============================================================
CAMPAIGN BRIEF
============================================================

{campaign_brief}


============================================================
ADDITIONAL POLICY CONTEXT
============================================================

{policy_display}


============================================================
ORIGINAL CONTENT
============================================================

{original_content}


============================================================
NON-BLOCKING ADVISORY FINDINGS
============================================================

{advisory_text}


============================================================
EDITING RULES
============================================================

1. OPTIONAL ONLY

Address useful Advisory Findings only when doing so improves the content.


2. PRESERVE FACTS

Preserve all correct factual information.


3. PRESERVE CREATOR VOICE

Preserve existing creator voice, first-person perspective and authentic style
as much as possible.


4. RESPECT ADVISORY PROVENANCE

SUPPLIED_CONTEXT means the Advisory is grounded in supplied source material.

GENERAL_HEURISTIC means it comes from general marketing/platform knowledge or
model judgment.

Do NOT treat GENERAL_HEURISTIC as an authoritative policy rule or mandatory
campaign requirement.


5. ADVISORY SUGGESTIONS ARE NOT SOURCE OF TRUTH

A free-form Advisory suggestion may itself be unsafe or overly specific.

If following a suggestion would require an unsupported claim, inferred benefit,
new creator experience or stronger promise:

IGNORE that unsafe part.

Use a safer conservative edit instead.


6. DO NOT INVENT

Never invent:

- ingredients
- specifications
- prices
- certifications
- clinical evidence
- research findings
- medical claims
- health outcomes
- customer reviews
- performance claims
- guarantees
- creator traits
- new creator experiences


7. NO UNSUPPORTED INFERENCE

Examples:

"lightweight" does not automatically mean "fast absorbing".

"10,000mAh" does not automatically mean "battery lasts all day".


8. PRESERVE QUALIFIERS

Do not strengthen:

- up to
- approximately
- may
- designed for


9. BRAND / POLICY PRIORITY

Explicit Brand restrictions and supplied policy rules always override optional
style optimization.


10. AVOID UNNECESSARY FULL REWRITES

Change only what is useful for the selected Advisory improvements.


============================================================
FINAL SELF-CHECK
============================================================

Before returning, silently verify:

1. No unsupported factual claim was introduced.
2. No new creator trait or experience was invented.
3. No factual qualifier was strengthened.
4. General heuristics were not treated as mandatory rules.
5. Existing creator voice was preserved where possible.
6. Changes were limited to useful optional improvements.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the optimized marketing content.

Do NOT explain changes.
Do NOT provide reasoning.
Do NOT provide edit notes.
Do NOT use Markdown fences.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.2,
    )


# =========================================================
# Backward-Compatible Wrapper
# =========================================================

def revise_content(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict,
    policy_context: str = "",
    model_key: str | None = None,
    apply_advisory: bool = True,
) -> str:
    """
    Backward-compatible revision interface.

    Priority:

    Compliance + Requirement
        -> one combined mandatory edit

    Compliance only
        -> Minimal Compliance Edit

    Requirement only
        -> Minimal Requirement Completion

    Advisory only
        -> optional quality optimization only when apply_advisory=True

    For the new Creator Review UI, use apply_advisory=False unless a Human
    explicitly clicks an Optimize action.
    """

    compliance_findings = (
        evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    requirement_findings = (
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    if (
        compliance_findings
        and requirement_findings
    ):
        return fix_mandatory_issues(
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            original_content=original_content,
            evaluation=evaluation,
            policy_context=policy_context,
            model_key=model_key,
        )

    if compliance_findings:
        return fix_compliance_issues(
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            original_content=original_content,
            evaluation=evaluation,
            policy_context=policy_context,
            model_key=model_key,
        )

    if requirement_findings:
        return complete_requirements(
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            original_content=original_content,
            evaluation=evaluation,
            policy_context=policy_context,
            model_key=model_key,
        )

    advisory_findings = (
        evaluation.get(
            "advisory_findings",
            [],
        )
        or []
    )

    if (
        advisory_findings
        and apply_advisory
    ):
        return optimize_quality(
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            original_content=original_content,
            evaluation=evaluation,
            policy_context=policy_context,
            model_key=model_key,
        )

    return original_content