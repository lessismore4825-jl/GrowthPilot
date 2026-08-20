from src.llm_client import generate_text


# =========================================================
# Helper: Format Compliance Findings
# =========================================================

def format_compliance_findings(
    findings: list,
) -> str:
    """
    Convert structured compliance findings
    into a readable prompt section.
    """

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

    return "\n\n".join(
        sections
    )


# =========================================================
# Helper: Format Advisory Findings
# =========================================================

def format_advisory_findings(
    findings: list,
) -> str:
    """
    Convert non-blocking advisory findings
    into a readable prompt section.
    """

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
""".strip()
        )

    return "\n\n".join(
        sections
    )


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

    Core principle:

    Fix the identified blocking compliance issues
    using the smallest necessary change.

    Do NOT turn compliance revision into a general
    copywriting rewrite.
    """

    findings = evaluation.get(
        "compliance_findings",
        [],
    )


    # =====================================================
    # Nothing to Fix
    # =====================================================

    if not findings:

        return original_content


    findings_text = (
        format_compliance_findings(
            findings
        )
    )


    policy_context = (
        policy_context
        or ""
    ).strip()


    policy_context_display = (
        policy_context
        if policy_context
        else
        "No additional external policy was provided."
    )


    prompt = f"""
You are a strict marketing compliance editor.

Your task is NOT to rewrite or improve the marketing copy.

Your task is ONLY to perform a:

MINIMAL COMPLIANCE EDIT

This means:

Fix every identified blocking compliance problem
using the smallest necessary textual change.

Preserve everything else whenever reasonably possible.


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

{policy_context_display}


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

Correct ONLY the identified compliance problems.

Do NOT treat this task as:

- creative rewriting
- style optimization
- platform optimization
- selling-point enrichment
- copy expansion
- engagement optimization


============================================================
MINIMAL EDIT RULES
============================================================

RULE 1 — MINIMIZE CHANGES

Make the smallest possible change needed
to remove each blocking compliance problem.

If one phrase is wrong,
change or remove that phrase.

Do not rewrite an entire sentence
when a smaller edit is sufficient.

Do not rewrite the entire content
unless the identified violations are so extensive
that a small edit cannot produce coherent content.


RULE 2 — PRESERVE CORRECT CONTENT

Preserve:

- correct facts
- existing sentence structure
- existing platform style
- existing tone
- existing selling points
- existing wording

whenever they are not responsible
for the blocking compliance problem.


RULE 3 — DO NOT ADD NEW MARKETING CLAIMS

Do NOT add any new claim merely to make
the revised content sound better.

Do NOT introduce new:

- product benefits
- product functions
- performance claims
- sensory claims
- emotional outcomes
- user outcomes
- lifestyle outcomes
- product positioning
- technical properties
- safety claims
- health claims
- medical claims
- superiority claims
- market leadership claims


RULE 4 — NO UNSUPPORTED ENRICHMENT

Do NOT infer one property from another.

Examples:

"lightweight"
does NOT automatically mean
"fast absorbing".

"non-greasy"
does NOT automatically mean
"non-sticky".

"designed for dry and sensitive skin"
does NOT automatically mean
"clinically suitable for sensitive skin".

"up to 24 hours hydration"
does NOT mean
"your skin will stay hydrated all day".

Do not add such inferred statements
unless they are explicitly supported.


RULE 5 — NO INVENTED EXPERIENCE

Do NOT invent first-person or user experiences.

Forbidden examples include:

"I tried it..."

"After using it..."

"My skin became..."

"It solved my..."

"One application made..."

unless that exact experience is explicitly
provided as approved source material.


RULE 6 — NO INVENTED EVIDENCE

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


RULE 7 — PRESERVE FACTUAL QUALIFIERS

Preserve uncertainty and qualification exactly.

For example:

"up to 24 hours"
must remain qualified.

Acceptable:
"up to 24 hours"
"最长可达24小时"

Not acceptable:
"24 hours guaranteed"
"一整天都保持水润"
"全天持续保湿无压力"


"approximately"
must not become an exact guarantee.


"may"
must not become
"will".


"designed for"
must not become
"clinically proven for".


RULE 8 — DO NOT STRENGTHEN CLAIMS

Never replace a factual statement
with a stronger marketing statement.

Examples:

Do NOT change:

"contains hyaluronic acid"

into:

"deeply locks moisture into the skin"


Do NOT change:

"lightweight texture"

into:

"instantly absorbs into the skin"


Do NOT change:

"designed for dry and sensitive skin"

into:

"repairs sensitive skin problems"


RULE 9 — BRAND AND POLICY RULES HAVE PRIORITY

If the Campaign Brief conflicts with:

- Brand Information
- explicit brand restrictions
- Additional Policy Context

the compliance-safe interpretation has priority.


RULE 10 — DO NOT FIX ADVISORY ISSUES

This task is ONLY for blocking compliance findings.

Do NOT proactively fix:

- platform style
- tone preferences
- creativity
- length preferences
- selling-point coverage
- engagement
- wording elegance

unless changing one of them is strictly necessary
to correct a blocking compliance issue.


============================================================
WHEN DELETION IS ENOUGH
============================================================

If removing the problematic wording produces
a coherent and usable sentence:

REMOVE IT.

Do not replace it with a new promotional claim
just to maintain the same length.


============================================================
WHEN REPLACEMENT IS NECESSARY
============================================================

If a false statement can be corrected using
an explicit supplied fact:

replace it directly with that supplied fact.

Example:

Original:
"72 hours of hydration"

Brand Information:
"up to 24 hours of hydration"

Preferred correction:
"up to 24 hours of hydration"

Do NOT expand it into additional claims.


============================================================
WHEN THE ORIGINAL IS MOSTLY INVALID
============================================================

If the blocking violations cover so much of
the original content that simple deletion would make
the content unusable:

create the smallest coherent replacement possible.

In that situation:

1. Use ONLY explicit facts from Brand Information.
2. Follow only necessary Campaign Brief requirements.
3. Add no unsupported benefits.
4. Add no inferred product characteristics.
5. Add no invented user experiences.
6. Add no unnecessary creative language.
7. Keep the replacement concise.


============================================================
FINAL SELF-CHECK
============================================================

Before returning the revision, silently verify:

1. Every blocking finding has been addressed.

2. No new factual claim was introduced
   unless directly supported by the supplied information.

3. No new personal experience was invented.

4. No new performance or sensory property was inferred.

5. No factual qualifier was strengthened.

6. Correct parts of the original were preserved
   whenever possible.

7. The revision changed only what was necessary
   for compliance.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the revised marketing content.

Do NOT explain what you changed.

Do NOT provide bullet points describing edits.

Do NOT provide reasoning.

Do NOT use Markdown fences.
"""


    return generate_text(
        prompt=prompt,
        model_key=model_key,

        # Compliance editing should be as
        # deterministic and conservative
        # as possible.
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
    Optional quality optimization.

    Unlike compliance repair, this action
    is not mandatory.

    It is triggered only when a human chooses
    to act on advisory findings.
    """

    advisory_findings = evaluation.get(
        "advisory_findings",
        [],
    )


    if not advisory_findings:

        return original_content


    advisory_text = (
        format_advisory_findings(
            advisory_findings
        )
    )


    policy_context = (
        policy_context
        or ""
    ).strip()


    policy_context_display = (
        policy_context
        if policy_context
        else
        "No additional external policy was provided."
    )


    prompt = f"""
You are a careful marketing content editor.

This task is an OPTIONAL QUALITY OPTIMIZATION.

Improve the content based on the supplied
non-blocking advisory findings.

Unlike compliance repair,
these are suggestions rather than mandatory violations.


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

{policy_context_display}


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

1. Address useful advisory findings
   when doing so improves the content.

2. Preserve all correct factual information.

3. Do not introduce unsupported product claims.

4. Do not invent:

   - ingredients
   - specifications
   - prices
   - certifications
   - clinical evidence
   - research findings
   - medical claims
   - health outcomes
   - user experiences
   - customer reviews
   - performance claims

5. Do not infer unsupported properties.

For example:

"lightweight"
does not automatically mean
"fast absorbing".

"non-greasy"
does not automatically mean
"non-sticky".

6. Do not strengthen factual qualifiers.

"up to"
must remain qualified.

"approximately"
must remain approximate.

"may"
must not become
"will".

7. Brand restrictions and supplied policy
   always override stylistic optimization.

8. Improve only what is useful.

9. Avoid unnecessary full rewrites.

10. Do not explain your changes.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the optimized marketing content.

Do NOT provide explanations.

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
) -> str:
    """
    Backward-compatible revision interface.

    Priority:

    Blocking compliance findings
        ↓
    Minimal Compliance Edit

    Otherwise:

    Advisory findings
        ↓
    Optional Quality Optimization

    This wrapper is retained so older code
    importing revise_content does not break.
    """

    compliance_findings = (
        evaluation.get(
            "compliance_findings",
            [],
        )
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


    advisory_findings = (
        evaluation.get(
            "advisory_findings",
            [],
        )
    )


    if advisory_findings:

        return optimize_quality(
            brand_info=brand_info,

            campaign_brief=campaign_brief,

            original_content=original_content,

            evaluation=evaluation,

            policy_context=policy_context,

            model_key=model_key,
        )


    # No issue found:
    # return the original without spending
    # another LLM request.
    return original_content