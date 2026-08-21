import re

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
# Optional Quality Optimization + Creator Provenance Guard
# =========================================================

# v2.3 targeted hardening:
#
# Held-out case H16 showed that a stylistic optimization could invent
# first-person creator usage / identity language even though the original
# creator draft did not contain that experience.
#
# The mandatory Compliance / Requirement editing paths above are intentionally
# unchanged. This section adds a narrow provenance guard only to the optional,
# human-triggered quality-optimization path.


_FIRST_PERSON_PATTERNS = (
    # Chinese
    r"我",
    r"我们",
    r"本人",

    # English
    r"\bI\b",
    r"\bI'm\b",
    r"\bI've\b",
    r"\bI’ve\b",
    r"\bI'd\b",
    r"\bI'll\b",
    r"\bmy\b",
    r"\bmine\b",
    r"\bme\b",
    r"\bpersonally\b",
)


# v2.3.1:
# Detect provenance at the SEMANTIC CATEGORY level instead of comparing
# exact phrases. This prevents false positives such as:
#
#   ORIGINAL: 最近试了一下
#   REVISED:  最近试用了
#
# Both express the same pre-existing creator trial experience, so this should
# not be treated as newly invented experience.
#
# At the same time, introducing a NEW category such as daily routine,
# creator identity, personal result, or ongoing usage is still flagged.

_CREATOR_EXPERIENCE_CATEGORY_PATTERNS = {
    "TRIAL_OR_PRIOR_USAGE": (
        # Chinese
        r"最近.{0,4}试(?:用)?(?:了|过|一下|试看)?",
        r"试(?:用)?(?:了|过|一下|下来)",
        r"用(?:了|过)(?:几天|几周|一段时间)",
        r"体验(?:了|过|下来)",

        # English
        r"\bi tried\b",
        r"\bi have tried\b",
        r"\bi've tried\b",
        r"\bafter using\b",
        r"\bafter trying\b",
    ),

    "ONGOING_USAGE": (
        # Chinese
        r"一直在用",
        r"最近在用",
        r"这几天在用",
        r"这段时间在用",
        r"最近.{0,4}一直.{0,4}用",

        # English
        r"\bbeen using\b",
        r"\bi am using\b",
        r"\bi'm using\b",
        r"\bi use this\b",
        r"\bi use it\b",
    ),

    "DAILY_OR_REPEATED_ROUTINE": (
        # Chinese
        r"每天都",
        r"每天会",
        r"每天用",
        r"平时我",
        r"我平时",
        r"日常我",
        r"我的日常",

        # English
        r"\bevery day\b",
        r"\bdaily routine\b",
        r"\bmy routine\b",
        r"\bi use .* every\b",
    ),

    "CREATOR_IDENTITY_OR_ROLE": (
        # Chinese
        r"我这种",
        r"像我这种",
        r"作为一个",
        r"作为一名",
        r"作为创作者",
        r"我是一个",
        r"我是做",
        r"我做(?:营销|运营|内容|设计|视频|摄影|写作)",

        # English
        r"\bas a creator\b",
        r"\bas a marketer\b",
        r"\bas a designer\b",
        r"\bas a writer\b",
        r"\bas someone who\b",
        r"\bfor someone like me\b",
    ),

    "PERSONAL_ATTRIBUTE": (
        # Chinese
        r"我的皮肤",
        r"我的肤质",
        r"我的头发",
        r"我的身体",
        r"我的生活",
        r"我的工作",
        r"我的团队",

        # English
        r"\bmy skin\b",
        r"\bmy hair\b",
        r"\bmy body\b",
        r"\bmy lifestyle\b",
        r"\bmy work\b",
        r"\bmy team\b",
        r"\bmy workflow\b",
    ),

    "PERSONAL_RESULT_OR_OUTCOME": (
        # Chinese
        r"对我来说",
        r"让我觉得",
        r"让我感觉",
        r"让我更",
        r"我觉得效果",
        r"我发现",
        r"我明显感觉",
        r"用了以后",
        r"用完之后",
        r"使用后",

        # English
        r"\bworks for me\b",
        r"\bworked for me\b",
        r"\bfor me personally\b",
        r"\bi noticed\b",
        r"\bi found that\b",
        r"\bafter using .* i\b",
    ),

    "PERSONAL_ENDORSEMENT": (
        # Chinese
        r"我很推荐",
        r"我会推荐",
        r"我推荐",
        r"我真的推荐",

        # English
        r"\bi recommend\b",
        r"\bi'd recommend\b",
        r"\bi would recommend\b",
    ),
}


def _content_origin_value(
    evaluation: dict,
) -> str:
    """Return normalized content origin."""

    return str(
        evaluation.get(
            "content_origin",
            "generated",
        )
        or "generated"
    ).strip().lower()


def _contains_first_person_marker(
    text: str,
) -> bool:
    """
    Conservative multilingual first-person smoke check.

    This is not semantic proof. It is only a deterministic fail-safe for
    obvious provenance regressions.
    """

    value = (
        text
        or ""
    )

    for pattern in _FIRST_PERSON_PATTERNS:

        if re.search(
            pattern,
            value,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def _creator_experience_categories(
    text: str,
) -> set[str]:
    """
    Return high-signal creator-experience semantic categories found in text.

    Categories are intentionally broader than exact phrases so harmless
    paraphrases of an already supplied creator experience are preserved.
    """

    value = (
        text
        or ""
    )

    found = set()

    for category, patterns in (
        _CREATOR_EXPERIENCE_CATEGORY_PATTERNS.items()
    ):

        for pattern in patterns:

            if re.search(
                pattern,
                value,
                flags=re.IGNORECASE,
            ):
                found.add(
                    category
                )

                break

    return found


def detect_creator_experience_provenance_risks(
    original_content: str,
    revised_content: str,
) -> list[str]:
    """
    Detect high-signal creator-experience categories introduced by revision.

    v2.3.1 behavior:
    - equivalent paraphrases within an already present experience category
      are NOT flagged;
    - genuinely new categories (e.g. daily routine, creator identity,
      personal result) ARE flagged;
    - new explicit first-person perspective is allowed only when the original
      already contained either first-person voice OR an implicit creator
      experience category.

    This is a deterministic regression / debugging guard. Semantic provenance
    validation is still performed by the separate repair pass.
    """

    original = (
        original_content
        or ""
    )

    revised = (
        revised_content
        or ""
    )

    original_categories = (
        _creator_experience_categories(
            original
        )
    )

    revised_categories = (
        _creator_experience_categories(
            revised
        )
    )

    risks = []

    # A new explicit first-person pronoun is not automatically a provenance
    # failure if the original already clearly expressed creator experience in
    # an implicit way (e.g. "最近试了一下").
    if (
        _contains_first_person_marker(
            revised
        )
        and not _contains_first_person_marker(
            original
        )
        and not original_categories
    ):
        risks.append(
            "NEW_FIRST_PERSON_PERSPECTIVE"
        )

    new_categories = (
        revised_categories
        - original_categories
    )

    for category in sorted(
        new_categories
    ):
        risks.append(
            f"NEW_CREATOR_EXPERIENCE_CATEGORY:{category}"
        )

    return risks


def _format_provenance_risks(
    risks: list[str],
) -> str:
    """Render deterministic provenance flags for the semantic repair pass."""

    if not risks:
        return (
            "No high-signal deterministic flag was detected. "
            "Still perform semantic comparison because unsupported creator "
            "experience may not match a simple phrase pattern."
        )

    return "\n".join(
        f"- {risk}"
        for risk in risks
    )


def _repair_creator_experience_provenance(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    optimized_candidate: str,
    advisory_text: str,
    policy_context: str,
    evaluation: dict,
    model_key: str | None = None,
) -> str:
    """
    Narrow post-optimization provenance repair.

    Only creator-submitted content receives this extra semantic comparison.
    It is not another creative rewrite.
    """

    if (
        _content_origin_value(
            evaluation
        )
        != "creator_draft"
    ):
        return optimized_candidate

    policy_display = _policy_context_display(
        policy_context
    )

    risk_flags = (
        detect_creator_experience_provenance_risks(
            original_content=
                original_content,

            revised_content=
                optimized_candidate,
        )
    )

    risk_text = _format_provenance_risks(
        risk_flags
    )

    prompt = f"""
You are a strict provenance-preservation editor.

This is NOT a creative rewrite.

Your task is ONLY to perform a:

CREATOR EXPERIENCE PROVENANCE CHECK

The ORIGINAL CONTENT was submitted by a creator.

The OPTIMIZED CANDIDATE was generated by AI after a Human user requested
optional quality optimization.

The ORIGINAL CONTENT is the ONLY source of truth for creator-specific
personal facts and experiences.


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
ORIGINAL CREATOR-SUBMITTED CONTENT
============================================================

{original_content}


============================================================
AI-OPTIMIZED CANDIDATE
============================================================

{optimized_candidate}


============================================================
OPTIONAL ADVISORY CONTEXT
============================================================

{advisory_text}


============================================================
DETERMINISTIC SMOKE-CHECK FLAGS
============================================================

{risk_text}


============================================================
PROVENANCE RULE
============================================================

The AI may preserve or lightly rephrase creator-specific personal meaning
ONLY when equivalent meaning already exists in ORIGINAL CONTENT.

The Campaign Brief / Creator Profile may describe desired style or creator
type.

It is NOT evidence that the actual creator personally:

- used the product
- used it for a certain duration
- uses it every day
- has a particular routine
- has a particular profession
- has a particular skin / hair / body trait
- experienced a particular outcome
- recommends the product
- identifies as a particular kind of creator

For example:

Creator Profile:
"Lifestyle video creator"

does NOT authorize the AI to invent:

"I've been using this for my daily vlogs"
"As a lifestyle creator, I..."
"This is perfect for creators like me"

unless equivalent personal meaning already appears in ORIGINAL CONTENT.


============================================================
REPAIR RULES
============================================================

1. Compare every creator-specific personal statement in the candidate against
   ORIGINAL CONTENT.

2. If equivalent personal meaning is absent from ORIGINAL CONTENT:
   REMOVE it or rewrite it into neutral, non-personal wording.

3. Preserve authentic first-person creator experience that already existed in
   ORIGINAL CONTENT.

4. Keep useful style improvements whenever possible:
   - shorter sentences
   - clearer structure
   - simpler wording
   - less formal wording
   - better flow
   - moderate tone adjustment

5. Do NOT add any new information during this repair pass.

6. Do NOT invent:
   - new product facts
   - benefits
   - performance claims
   - guarantees
   - creator traits
   - creator identity
   - usage history
   - usage duration
   - routine
   - personal results
   - endorsements

7. If uncertain whether a creator-specific statement is supported by the
   ORIGINAL CONTENT, remove the personal claim and prefer neutral wording.

8. Brand Information remains authoritative for product facts.
   ORIGINAL CONTENT is authoritative only for creator-specific experience
   already expressed there.

9. PRESERVE EPISTEMIC STRENGTH

Do not convert a creator's subjective or qualified observation into a stronger
objective product-performance claim.

Examples:

Original:
"报销分类真的省事很多"

Safer paraphrase:
"用起来会更省事一些"

Avoid:
"能大幅简化报销管理工作"

Original:
"挺方便"

Safer paraphrase:
"使用上比较方便"

Avoid:
"显著提升效率"

Do not add stronger words such as:

- 大幅
- 显著
- 完全
- 保证
- 一定
- dramatically
- significantly
- completely
- guaranteed

unless equivalent strength already exists in ORIGINAL CONTENT or an explicit
supplied factual source.


============================================================
FINAL SELF-CHECK
============================================================

Before returning, silently verify:

1. Every creator-specific personal statement is supported by equivalent
   meaning in ORIGINAL CONTENT.
2. No new usage history or duration was invented.
3. No new creator identity or trait was invented.
4. No new personal result or endorsement was invented.
5. Safe style improvements were preserved where possible.
6. No new product claim was introduced during repair.
7. Existing subjective or qualified creator statements were not strengthened
   into broader objective performance claims.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the provenance-safe optimized marketing content.

Do NOT explain changes.
Do NOT provide reasoning.
Do NOT provide edit notes.
Do NOT use Markdown fences.
"""

    repaired = generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.0,
    )

    # Last-resort deterministic fail-safe:
    #
    # If the creator's original draft had no first-person voice at all but the
    # provenance-repaired output still introduces first-person voice, optional
    # optimization is rejected and the original draft is returned.
    #
    # Optional optimization is non-mandatory, so no optimization is safer than
    # unsupported creator testimony.
    original_experience_categories = (
        _creator_experience_categories(
            original_content
        )
    )

    if (
        not _contains_first_person_marker(
            original_content
        )
        and not original_experience_categories
        and _contains_first_person_marker(
            repaired
        )
    ):
        return original_content

    return repaired


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

    v2.3.1 provenance hardening:
    - Creator Profile is treated as style context, not personal evidence.
    - New creator experiences / identities are explicitly prohibited.
    - Creator drafts receive a second semantic provenance repair pass.
    - A deterministic first-person fail-safe remains as a last resort.
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

    content_origin = _content_origin_value(
        evaluation
    )

    if content_origin == "creator_draft":

        creator_provenance_instruction = """
The ORIGINAL CONTENT is creator-submitted.

Use the Creator Profile only to guide tone, format, pacing and style.

Do NOT use the Creator Profile as evidence about the creator's real identity,
habits, profession, lifestyle, product usage or personal experience.

Preserve first-person experience already present in ORIGINAL CONTENT.

Do NOT create NEW personal testimony such as:

- "I have been using this recently..."
- "I use this every day..."
- "After trying it..."
- "For me..."
- "As a creator..."
- "This is perfect for creators like me..."
- "My skin / my hair / my workflow..."
- any new personal result or endorsement

unless equivalent personal meaning already exists in ORIGINAL CONTENT.

If the original contains no personal experience, improve creator fit through
wording, rhythm, sentence structure and formatting WITHOUT fabricating a
personal story.
""".strip()

    else:

        creator_provenance_instruction = """
Do not invent first-person user experience, endorsements, personal outcomes or
personal traits while optimizing the content.
""".strip()

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
CREATOR EXPERIENCE PROVENANCE
============================================================

{creator_provenance_instruction}


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


3. PRESERVE CREATOR VOICE, BUT DO NOT INVENT CREATOR TESTIMONY

Preserve existing creator voice, first-person perspective and authentic style
as much as possible.

Distinguish:

A. Personal meaning already present in ORIGINAL CONTENT
   -> may be preserved or lightly rephrased.

B. Personal meaning created only during this AI rewrite
   -> NOT allowed when it asserts a new creator fact, identity, usage history,
      routine, preference, result, endorsement or experience.

A creator-style caption does NOT require invented first-person testimony.


4. CREATOR PROFILE IS STYLE CONTEXT, NOT PERSONAL EVIDENCE

A Creator Profile may guide:

- tone
- complexity
- format
- pacing
- level of formality

It may NOT be used as evidence to claim:

- the creator personally used the product
- the creator used it for a certain duration
- the creator has a particular lifestyle
- the creator has a particular profession
- the creator has a particular skin / hair / body trait
- the creator experienced a particular result


5. RESPECT ADVISORY PROVENANCE

SUPPLIED_CONTEXT means the Advisory is grounded in supplied source material.

GENERAL_HEURISTIC means it comes from general marketing/platform knowledge or
model judgment.

Do NOT treat GENERAL_HEURISTIC as an authoritative policy rule or mandatory
campaign requirement.


6. ADVISORY SUGGESTIONS ARE NOT SOURCE OF TRUTH

A free-form Advisory suggestion may itself be unsafe or overly specific.

If following a suggestion would require:

- unsupported factual claims
- inferred benefits
- new creator experiences
- new creator identities
- new personal outcomes
- stronger promises

IGNORE that unsafe part.

Use a safer conservative edit instead.


7. DO NOT INVENT

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
- creator identities
- creator routines
- creator usage history
- creator usage duration
- creator endorsements
- creator experiences
- creator outcomes


8. NO UNSUPPORTED INFERENCE

Examples:

"lightweight" does not automatically mean "fast absorbing".

"10,000mAh" does not automatically mean "battery lasts all day".

"Lifestyle creator" does not mean the creator personally uses the product for
daily lifestyle content.

"Professional creator" does not mean the creator personally uses the product
for professional work.


9. PRESERVE QUALIFIERS

Do not strengthen:

- up to
- approximately
- may
- designed for


10. BRAND / POLICY PRIORITY

Explicit Brand restrictions and supplied policy rules always override optional
style optimization.


11. AVOID UNNECESSARY FULL REWRITES

Change only what is useful for the selected Advisory improvements.

Do not replace an acceptable creator voice with a newly invented persona.


12. SAFE CREATOR-LIKE STYLE WHEN ORIGINAL HAS NO PERSONAL EXPERIENCE

If the content needs to feel:

- more natural
- more approachable
- more creator-like
- more platform-appropriate

and ORIGINAL CONTENT contains no first-person product experience:

use:

- shorter sentences
- clearer structure
- more conversational but non-personal wording
- cleaner emphasis
- simpler phrasing

Do NOT solve the style problem by fabricating personal usage.


13. DO NOT STRENGTHEN SUBJECTIVE LANGUAGE INTO OBJECTIVE PERFORMANCE CLAIMS

Optional style optimization may rephrase an existing creator observation, but
must preserve its epistemic strength.

Example:

Original:
"报销分类真的省事很多"

Acceptable:
"报销分类用起来更省事一些"

Avoid:
"能大幅简化报销管理工作"

Original:
"挺方便"

Acceptable:
"使用上比较方便"

Avoid:
"显著提升效率"

Do not introduce stronger outcome language such as:

- 大幅
- 显著
- 完全
- 保证
- 一定
- dramatically
- significantly
- completely
- guaranteed

unless equivalent strength is explicitly supported by ORIGINAL CONTENT or
another supplied factual source.


============================================================
FINAL SELF-CHECK
============================================================

Before returning, silently verify:

1. No unsupported factual claim was introduced.
2. No new creator trait, identity, routine or experience was invented.
3. Every first-person product experience in the output is supported by
   equivalent meaning in ORIGINAL CONTENT.
4. Creator Profile was used only for style guidance.
5. No factual qualifier was strengthened.
6. General heuristics were not treated as mandatory rules.
7. Existing creator voice was preserved where possible.
8. Existing subjective or qualified observations were not strengthened into
   broader objective performance claims.
9. Changes were limited to useful optional improvements.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY the optimized marketing content.

Do NOT explain changes.
Do NOT provide reasoning.
Do NOT provide edit notes.
Do NOT use Markdown fences.
"""

    optimized_candidate = generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.2,
    )

    return _repair_creator_experience_provenance(
        brand_info=brand_info,
        campaign_brief=campaign_brief,
        original_content=original_content,
        optimized_candidate=optimized_candidate,
        advisory_text=advisory_text,
        policy_context=policy_context,
        evaluation=evaluation,
        model_key=model_key,
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