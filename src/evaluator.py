import re
import unicodedata

from src.llm_client import (
    JUDGE_MODEL_KEY,
    generate_text,
)


# =========================================================
# Diagnostic Score Fields
# =========================================================

SCORE_FIELDS = [
    "brand_alignment",
    "tone_match",
    "selling_point_coverage",
    "factual_consistency",
    "unsupported_claim_risk",
]


# =========================================================
# Allowed Policy Sources
# =========================================================

# These are not issue categories.
#
# They define the only supplied sources that may
# support a blocking compliance decision.

ALLOWED_POLICY_SOURCES = {
    "BRAND INFORMATION",
    "CAMPAIGN BRIEF",
    "ADDITIONAL POLICY CONTEXT",
}


# =========================================================
# Response Cleaning
# =========================================================

def clean_response_text(
    response_text: str,
) -> str:
    """
    Clean common LLM formatting artifacts.

    The evaluator intentionally does not depend
    on JSON output.
    """

    if not isinstance(
        response_text,
        str,
    ):
        raise ValueError(
            "Judge response must be a string."
        )

    cleaned = (
        response_text
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

    # Remove hidden reasoning blocks when present.
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    # Remove accidental Markdown fences.
    cleaned = re.sub(
        r"```(?:text|txt|json|python)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    )

    return cleaned.strip()


# =========================================================
# Key-Value Parser
# =========================================================

def parse_key_value_response(
    response_text: str,
) -> dict:
    """
    Parse simple line-based Judge output.

    Example:

    BRAND_ALIGNMENT=8
    TONE_MATCH=7

    Values may contain quotation marks,
    colons, commas and Chinese punctuation.

    Parsing splits only on the first "=".
    """

    cleaned = clean_response_text(
        response_text
    )

    if not cleaned:
        raise ValueError(
            "Judge returned an empty response."
        )

    parsed = {}

    for raw_line in cleaned.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.upper() in {
            "BEGIN_EVALUATION",
            "END_EVALUATION",
            "BEGIN_PAIRWISE",
            "END_PAIRWISE",
        }:
            continue

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1,
        )

        key = (
            key.strip()
            .upper()
            .replace(" ", "_")
            .strip("\"'")
        )

        value = value.strip()

        # Some models partially imitate JSON.
        if value.endswith(","):
            value = value[:-1].rstrip()

        # Remove one matching outer quote pair.
        if len(value) >= 2:

            quote_pairs = [
                ('"', '"'),
                ("'", "'"),
                ("“", "”"),
                ("‘", "’"),
            ]

            for left, right in quote_pairs:

                if (
                    value.startswith(left)
                    and value.endswith(right)
                ):
                    value = value[
                        len(left):
                        len(value) - len(right)
                    ].strip()

                    break

        if key:
            parsed[
                key
            ] = value

    if not parsed:

        preview = repr(
            cleaned[:500]
        )

        raise ValueError(
            "Judge response could not be parsed "
            "as key-value output. "
            f"Raw response preview: {preview}"
        )

    return parsed


# =========================================================
# Score Normalization
# =========================================================

def normalize_score(
    value,
    field_name: str,
):
    """
    Normalize a diagnostic score to 1-10.
    """

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        # Supports:
        #
        # 8
        # 8.0
        # 8/10

        score_match = re.match(
            r"^-?\d+(?:\.\d+)?",
            value,
        )

        if not score_match:

            raise ValueError(
                f"{field_name} must be numeric. "
                f"Received: {value}"
            )

        value = float(
            score_match.group()
        )

    if not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    if not 1 <= value <= 10:
        raise ValueError(
            f"{field_name} must be between "
            f"1 and 10."
        )

    if float(
        value
    ).is_integer():

        return int(
            value
        )

    return round(
        float(
            value
        ),
        2,
    )


# =========================================================
# Policy Source Normalization
# =========================================================

def normalize_policy_source(
    value: str,
) -> str:
    """
    Normalize common source-name variations.

    This is a closed source list because these
    are the only documents supplied to the Judge.

    This does NOT impose a closed issue taxonomy.
    """

    text = str(
        value
        or ""
    ).strip()

    normalized = (
        text
        .upper()
        .replace("_", " ")
        .replace("-", " ")
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    aliases = {
        "BRAND INFO":
            "BRAND INFORMATION",

        "BRAND":
            "BRAND INFORMATION",

        "BRAND INFORMATION":
            "BRAND INFORMATION",

        "CAMPAIGN":
            "CAMPAIGN BRIEF",

        "BRIEF":
            "CAMPAIGN BRIEF",

        "CAMPAIGN BRIEF":
            "CAMPAIGN BRIEF",

        "POLICY CONTEXT":
            "ADDITIONAL POLICY CONTEXT",

        "ADDITIONAL POLICY":
            "ADDITIONAL POLICY CONTEXT",

        "EXTERNAL POLICY":
            "ADDITIONAL POLICY CONTEXT",

        "ADDITIONAL POLICY CONTEXT":
            "ADDITIONAL POLICY CONTEXT",
    }

    return aliases.get(
        normalized,
        normalized,
    )


# =========================================================
# Indexed Field Helpers
# =========================================================

def get_indexed_numbers(
    parsed: dict,
    prefix: str,
) -> list:
    """
    Find indexes such as:

    COMPLIANCE_1_EVIDENCE
    COMPLIANCE_2_REASON

    -> [1, 2]
    """

    indexes = set()

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)_"
    )

    for key in parsed:

        match = pattern.match(
            key
        )

        if match:

            indexes.add(
                int(
                    match.group(
                        1
                    )
                )
            )

    return sorted(
        indexes
    )


# =========================================================
# Grounding Helpers
# =========================================================

def strip_wrapping_quotes(
    text: str,
) -> str:
    """
    Remove one matching outer quotation pair.

    Internal quotation marks are preserved.
    """

    value = str(
        text
        or ""
    ).strip()

    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
    ]

    for left, right in quote_pairs:

        if (
            value.startswith(left)
            and value.endswith(right)
            and len(value)
            >= len(left) + len(right)
        ):

            return value[
                len(left):
                len(value) - len(right)
            ].strip()

    return value


def normalize_for_grounding(
    text: str,
) -> str:
    """
    Conservative normalization used for
    deterministic grounding checks.

    We normalize:

    - Unicode width
    - hidden characters
    - repeated whitespace
    - case

    We intentionally preserve punctuation
    and semantic content.
    """

    value = str(
        text
        or ""
    )

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = (
        value
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value.casefold()


def text_is_grounded(
    quoted_text: str,
    source_text: str,
) -> bool:
    """
    Check whether a quoted Evidence / Policy Basis
    actually appears in its claimed source.

    This is intentionally stricter than semantic
    similarity.

    The purpose is provenance validation,
    not semantic classification.

    A paraphrase is NOT enough for a hard
    blocking decision.
    """

    quote = strip_wrapping_quotes(
        quoted_text
    )

    source = str(
        source_text
        or ""
    )

    if not quote or not source:
        return False

    normalized_quote = (
        normalize_for_grounding(
            quote
        )
    )

    normalized_source = (
        normalize_for_grounding(
            source
        )
    )

    if not normalized_quote:
        return False

    # Standard substring check.
    if normalized_quote in normalized_source:
        return True

    # Allow differences caused only by line breaks
    # or spaces, especially common in Chinese text.
    compact_quote = re.sub(
        r"\s+",
        "",
        normalized_quote,
    )

    compact_source = re.sub(
        r"\s+",
        "",
        normalized_source,
    )

    if (
        compact_quote
        and compact_quote
        in compact_source
    ):
        return True

    return False


def get_policy_source_text(
    policy_source: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
) -> str:
    """
    Return the actual text belonging to
    a declared policy source.
    """

    source_map = {
        "BRAND INFORMATION":
            brand_info,

        "CAMPAIGN BRIEF":
            campaign_brief,

        "ADDITIONAL POLICY CONTEXT":
            policy_context,
    }

    return str(
        source_map.get(
            policy_source,
            "",
        )
        or ""
    )


def is_yes(
    value: str,
) -> bool:
    """
    Normalize a direct-conflict flag.
    """

    normalized = str(
        value
        or ""
    ).strip().upper()

    return normalized in {
        "YES",
        "Y",
        "TRUE",
        "1",
    }


# =========================================================
# Grounding Downgrade Helper
# =========================================================

def make_grounding_advisory(
    evidence: str,
    reason: str,
) -> dict:
    """
    Preserve a potential concern when the
    deterministic grounding gate rejects it
    as a blocking finding.

    This avoids silently throwing away
    potentially useful human-review signals.
    """

    return {
        "area":
            "Grounding Review",

        "evidence":
            evidence,

        "reason":
            reason,

        "suggestion":
            (
                "Review this concern manually. "
                "It was not treated as a blocking "
                "compliance finding because the "
                "required deterministic grounding "
                "could not be verified."
            ),
    }


# =========================================================
# Compliance Finding Parser + Deterministic Grounding Gate
# =========================================================

def parse_compliance_findings(
    parsed: dict,
    generated_content: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
) -> tuple[list, list, list]:
    """
    Parse and validate proposed blocking findings.

    A finding becomes BLOCKING only when ALL
    deterministic gates pass.

    Required gates:

    1. The Judge explicitly marks it as
       a DIRECT_CONFLICT.

    2. Evidence is an actual quotation from
       GENERATED CONTENT.

    3. Policy Source is one of the supplied
       source documents.

    4. Policy Basis is an actual quotation
       from that exact Policy Source.

    5. If Additional Policy Context is cited,
       such context must actually have been supplied.

    Findings that fail the grounding gate are
    downgraded to Advisory rather than silently removed.

    Returns:

    (
        valid_blocking_findings,
        parser_review_notes,
        downgraded_advisories
    )
    """

    findings = []

    review_notes = []

    downgraded_advisories = []

    indexes = get_indexed_numbers(
        parsed,
        "COMPLIANCE",
    )

    for index in indexes:

        prefix = (
            f"COMPLIANCE_{index}"
        )

        direct_conflict = parsed.get(
            f"{prefix}_DIRECT_CONFLICT",
            "",
        ).strip()

        evidence = parsed.get(
            f"{prefix}_EVIDENCE",
            "",
        ).strip()

        policy_source = (
            normalize_policy_source(
                parsed.get(
                    f"{prefix}_POLICY_SOURCE",
                    "",
                )
            )
        )

        policy_basis = parsed.get(
            f"{prefix}_POLICY_BASIS",
            "",
        ).strip()

        reason = parsed.get(
            f"{prefix}_REASON",
            "",
        ).strip()

        required_action = parsed.get(
            f"{prefix}_REQUIRED_ACTION",
            "",
        ).strip()


        # =================================================
        # Gate 0:
        # Required structure
        # =================================================

        if not (
            evidence
            and policy_source
            and policy_basis
            and reason
        ):

            message = (
                f"Potential compliance finding "
                f"{index} was not treated as blocking "
                f"because required grounding fields "
                f"were incomplete."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

            continue


        # =================================================
        # Gate 1:
        # Direct Conflict Only
        # =================================================

        if not is_yes(
            direct_conflict
        ):

            message = (
                f"Potential compliance finding "
                f"{index} was downgraded because the "
                f"Judge did not confirm a direct "
                f"text-to-rule conflict."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

            continue


        # =================================================
        # Gate 2:
        # Allowed Policy Source
        # =================================================

        if (
            policy_source
            not in ALLOWED_POLICY_SOURCES
        ):

            message = (
                f"Potential compliance finding "
                f"{index} referenced unsupported "
                f"policy source: {policy_source}. "
                f"It was not treated as blocking."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

            continue


        # =================================================
        # Gate 3:
        # Additional Policy Context must exist
        # =================================================

        if (
            policy_source
            == "ADDITIONAL POLICY CONTEXT"
            and not str(
                policy_context
                or ""
            ).strip()
        ):

            message = (
                f"Potential compliance finding "
                f"{index} cited ADDITIONAL POLICY "
                f"CONTEXT even though no additional "
                f"policy context was supplied."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

            continue


        # =================================================
        # Gate 4:
        # Evidence must exist in Generated Content
        # =================================================

        if not text_is_grounded(
            quoted_text=evidence,
            source_text=generated_content,
        ):

            message = (
                f"Potential compliance finding "
                f"{index} was downgraded because its "
                f"Evidence could not be verified as "
                f"an actual quotation from the "
                f"Generated Content."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

            continue


        # =================================================
        # Gate 5:
        # Policy Basis must exist in claimed source
        # =================================================

        source_text = get_policy_source_text(
            policy_source=policy_source,
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            policy_context=policy_context,
        )


        if not text_is_grounded(
            quoted_text=policy_basis,
            source_text=source_text,
        ):

            message = (
                f"Potential compliance finding "
                f"{index} was downgraded because its "
                f"Policy Basis could not be verified "
                f"in the claimed Policy Source "
                f"({policy_source})."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

            continue


        # =================================================
        # Finding Passed All Grounding Gates
        # =================================================

        findings.append(
            {
                "evidence":
                    strip_wrapping_quotes(
                        evidence
                    ),

                "policy_source":
                    policy_source,

                "policy_basis":
                    strip_wrapping_quotes(
                        policy_basis
                    ),

                "reason":
                    reason,

                "required_action":
                    required_action,
            }
        )


    return (
        findings,
        review_notes,
        downgraded_advisories,
    )


# =========================================================
# Advisory Finding Parser
# =========================================================

def parse_advisory_findings(
    parsed: dict,
) -> list:
    """
    Parse open-ended quality / risk suggestions.

    No closed issue taxonomy is imposed.
    """

    findings = []

    indexes = get_indexed_numbers(
        parsed,
        "ADVISORY",
    )

    for index in indexes:

        prefix = (
            f"ADVISORY_{index}"
        )

        area = parsed.get(
            f"{prefix}_AREA",
            "General",
        ).strip()

        evidence = parsed.get(
            f"{prefix}_EVIDENCE",
            "",
        ).strip()

        reason = parsed.get(
            f"{prefix}_REASON",
            "",
        ).strip()

        suggestion = parsed.get(
            f"{prefix}_SUGGESTION",
            "",
        ).strip()

        if not reason:
            continue

        findings.append(
            {
                "area":
                    area or "General",

                "evidence":
                    evidence,

                "reason":
                    reason,

                "suggestion":
                    suggestion,
            }
        )

    return findings


# =========================================================
# Human Review Notes Parser
# =========================================================

def parse_review_notes(
    parsed: dict,
) -> list:
    """
    Parse:

    REVIEW_NOTE_1=...
    REVIEW_NOTE_2=...
    """

    notes = []

    pattern = re.compile(
        r"^REVIEW_NOTE_(\d+)$"
    )

    indexed_notes = []

    for key, value in parsed.items():

        match = pattern.match(
            key
        )

        if not match:
            continue

        note = str(
            value
        ).strip()

        if note:

            indexed_notes.append(
                (
                    int(
                        match.group(
                            1
                        )
                    ),
                    note,
                )
            )

    indexed_notes.sort(
        key=lambda item: item[0]
    )

    for _, note in indexed_notes:

        notes.append(
            note
        )

    return notes


# =========================================================
# Heuristic Composite Score
# =========================================================

def calculate_heuristic_score(
    evaluation: dict,
) -> float:
    """
    Comparative diagnostic signal only.

    IMPORTANT:

    This score is NOT a calibrated
    acceptance threshold.

    It is used only for:

    - descriptive comparison
    - model benchmarking
    - V1 / V2 trend analysis
    """

    claim_safety = (
        11
        - evaluation[
            "unsupported_claim_risk"
        ]
    )

    score = (
        evaluation[
            "brand_alignment"
        ] * 0.20

        + evaluation[
            "tone_match"
        ] * 0.15

        + evaluation[
            "selling_point_coverage"
        ] * 0.20

        + evaluation[
            "factual_consistency"
        ] * 0.30

        + claim_safety * 0.15
    )

    return round(
        score,
        1,
    )


# =========================================================
# Evaluation Response Parser
# =========================================================

def parse_evaluation_response(
    response_text: str,
    generated_content: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
) -> dict:
    """
    Parse the line-based Judge response into
    the standard GrowthPilot evaluation schema.

    Blocking findings pass through the
    deterministic grounding gate before being
    accepted.
    """

    parsed = parse_key_value_response(
        response_text
    )

    score_mapping = {
        "brand_alignment":
            "BRAND_ALIGNMENT",

        "tone_match":
            "TONE_MATCH",

        "selling_point_coverage":
            "SELLING_POINT_COVERAGE",

        "factual_consistency":
            "FACTUAL_CONSISTENCY",

        "unsupported_claim_risk":
            "UNSUPPORTED_CLAIM_RISK",
    }

    evaluation = {}

    for field_name, response_key in (
        score_mapping.items()
    ):

        if response_key not in parsed:

            preview = repr(
                clean_response_text(
                    response_text
                )[:500]
            )

            raise ValueError(
                f"Judge response is missing "
                f"{response_key}. "
                f"Raw response preview: {preview}"
            )

        evaluation[
            field_name
        ] = normalize_score(
            parsed[
                response_key
            ],
            field_name,
        )


    (
        compliance_findings,
        grounding_review_notes,
        downgraded_advisories,
    ) = parse_compliance_findings(
        parsed=parsed,
        generated_content=generated_content,
        brand_info=brand_info,
        campaign_brief=campaign_brief,
        policy_context=policy_context,
    )


    model_advisories = (
        parse_advisory_findings(
            parsed
        )
    )


    model_review_notes = (
        parse_review_notes(
            parsed
        )
    )


    evaluation[
        "compliance_findings"
    ] = compliance_findings


    # Grounding-gate rejections are preserved
    # as advisory concerns.
    evaluation[
        "advisory_findings"
    ] = (
        model_advisories
        + downgraded_advisories
    )


    evaluation[
        "review_notes"
    ] = (
        model_review_notes
        + grounding_review_notes
    )


    evaluation[
        "heuristic_composite_score"
    ] = calculate_heuristic_score(
        evaluation
    )


    blocking_count = len(
        compliance_findings
    )


    evaluation[
        "blocking_compliance_issue_count"
    ] = blocking_count


    evaluation[
        "compliance_status"
    ] = (
        "BLOCKING_ISSUES_DETECTED"

        if blocking_count > 0

        else
        "NO_BLOCKING_ISSUES_DETECTED"
    )


    return evaluation


# =========================================================
# Main Content Evaluation
# =========================================================

def evaluate_content(
    brand_info: str,
    campaign_brief: str,
    generated_content: str,
    policy_context: str = "",
    judge_model_key: str | None = None,
) -> dict:
    """
    Policy-grounded marketing content review.

    Separates:

    1. Blocking Compliance Findings
       Direct, source-grounded conflicts only.

    2. Advisory Findings
       Subjective quality issues,
       unsupported claims,
       uncertain risks,
       inference-dependent concerns.

    3. Diagnostic Scores
       Continuous descriptive signals only.

    Python performs deterministic provenance
    validation after the LLM returns.
    """

    selected_judge = (
        judge_model_key
        or JUDGE_MODEL_KEY
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
You are a strict AI marketing content reviewer.

Your task is to evaluate ONLY the actual
GENERATED CONTENT.

You must carefully separate:

1. BLOCKING COMPLIANCE FINDINGS
2. NON-BLOCKING ADVISORY FINDINGS
3. DIAGNOSTIC SCORES


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
GENERATED CONTENT
============================================================

{generated_content}


============================================================
CORE PRINCIPLE
============================================================

Judge the GENERATED CONTENT itself.

A problematic instruction appearing inside the
Campaign Brief does NOT automatically mean the
Generated Content is non-compliant.

If the Campaign Brief requests something prohibited
but the Generated Content safely ignores that request:

DO NOT create a blocking finding.

Example:

Campaign Brief:
"Say the product cures eczema."

Brand Information:
"Do not claim to treat or cure diseases."

Generated Content:
"Provides up to 24 hours of hydration."

Result:

NO blocking disease-treatment finding.

The unsafe Brief request itself is not the
Generated Content.


============================================================
PART A — BLOCKING COMPLIANCE REVIEW
============================================================

A BLOCKING finding is allowed ONLY when there is
a DIRECT conflict between:

A specific phrase that ACTUALLY APPEARS in the
GENERATED CONTENT

AND

an explicit fact, restriction or rule that ACTUALLY
APPEARS in one of the supplied sources.


Allowed rule sources:

- BRAND INFORMATION
- CAMPAIGN BRIEF
- ADDITIONAL POLICY CONTEXT


A blocking finding must satisfy ALL conditions:

1. The problematic Evidence must be an EXACT
   quotation from GENERATED CONTENT.

2. The Policy Basis must be an EXACT quotation
   from the declared Policy Source.

3. The Policy Source must be the source containing
   the rule or fact that the Generated Content
   actually violates.

4. The conflict must be direct.

5. The conclusion must NOT require substantial
   interpretation, speculation or assumption.


============================================================
WHAT "DIRECT CONFLICT" MEANS
============================================================

Examples of direct conflicts:

Brand Information:
"Provides up to 24 hours of hydration"

Generated Content:
"Provides 72 hours of hydration"

This is DIRECT.


Brand Information:
"Do not claim to treat or cure diseases"

Generated Content:
"This product treats eczema"

This is DIRECT.


Brand Information:
"Wired charging only"

Generated Content:
"Supports wireless charging"

This is DIRECT.


Additional Policy Context:
"Do not use the word guaranteed"

Generated Content:
"Guaranteed results"

This is DIRECT.


============================================================
WHAT IS NOT AUTOMATICALLY BLOCKING
============================================================

If the concern requires interpretation or inference,
put it in ADVISORY instead.

Examples:

Generated Content:
"My redness improved a lot"

Supplied rule:
"Do not claim to cure diseases"

Do NOT automatically call this a disease-treatment
violation unless the generated wording itself
explicitly makes that prohibited treatment or cure
claim.

It may still be:

- an unsupported efficacy claim
- an authenticity concern
- a potentially sensitive claim

Those belong in ADVISORY when there is no direct
text-to-rule conflict.


Generated Content:
"I personally tried it and loved it"

Supplied rule:
"Do not guarantee medical results"

Do NOT automatically treat personal satisfaction
language as a guaranteed medical result.

It may still create authenticity or support risk,
which belongs in ADVISORY.


Generated Content:
"absorbs instantly"

Brand Information:
"lightweight"

Do NOT infer that lightweight proves or disproves
instant absorption.

If no explicit supplied rule prohibits the claim
and no supplied fact directly contradicts it,
treat the concern as ADVISORY.


============================================================
UNSUPPORTED CLAIMS
============================================================

Unsupported claims may be important.

Examples:

- "absorbs instantly"
- "will not pill under makeup"
- "no irritating additives"
- "redness improved"
- "my skin barrier became much stronger"
- invented personal experience

However:

UNSUPPORTED
does NOT automatically mean
BLOCKING COMPLIANCE VIOLATION.

If the supplied materials do not contain an
explicit rule or fact directly violated by the claim:

put it in ADVISORY.

The unsupported claim risk score may also be high.


============================================================
POLICY SOURCE RULE
============================================================

POLICY_SOURCE means:

THE SOURCE CONTAINING THE RULE OR FACT
THAT WAS VIOLATED.

It does NOT mean:

- the source that caused the model to make the claim
- the source containing a bad request
- the source providing campaign motivation


Example:

CAMPAIGN BRIEF:
"Say the product cures sensitive skin problems."

BRAND INFORMATION:
"Do not claim to treat or cure diseases."

GENERATED CONTENT:
"This product cures eczema."

The violated rule comes from:

BRAND INFORMATION

Therefore:

POLICY_SOURCE=BRAND INFORMATION

NOT:

POLICY_SOURCE=CAMPAIGN BRIEF


============================================================
CAMPAIGN BRIEF CONFLICTS
============================================================

Brand restrictions and supplied policy restrictions
take priority over conflicting creative requests
inside the Campaign Brief.

A Campaign Brief can only be used as a blocking
Policy Source when:

1. It contains an explicit mandatory factual or
   operational requirement,

AND

2. The Generated Content directly violates that
   requirement,

AND

3. That requirement does not conflict with higher
   priority Brand Information or supplied policy.

Do NOT use a prohibited campaign request as the
policy basis for declaring the final content unsafe.


============================================================
EVIDENCE RULE
============================================================

EVIDENCE must contain ONLY the exact problematic
wording from GENERATED CONTENT.

Do NOT include:

- Campaign Brief text
- Brand Information text
- explanations
- parenthetical comments
- your interpretation
- multiple sources joined by "/"
- paraphrased content

Bad:

EVIDENCE=Campaign asks for cure / user says redness improved

Good:

EVIDENCE=之前动不动就泛红刺痛的情况真的少了好多


============================================================
POLICY BASIS RULE
============================================================

POLICY_BASIS must be copied directly from the
declared Policy Source.

Do NOT paraphrase.

Do NOT combine multiple different rules into a
new sentence.

Do NOT explain the rule inside POLICY_BASIS.

If one piece of content directly violates multiple
rules, either:

- choose the clearest governing rule

or

- create separate findings.


Example:

Brand Information contains:

Do not claim to treat or cure diseases

Then use exactly:

POLICY_BASIS=Do not claim to treat or cure diseases

Do NOT write:

POLICY_BASIS=The brand prohibits medical treatment
claims and disease claims.


============================================================
DIRECT CONFLICT FLAG
============================================================

Every proposed blocking finding must include:

DIRECT_CONFLICT=YES

ONLY use YES when:

- the Evidence is in Generated Content
- the Policy Basis is in the stated source
- the wording directly conflicts with that rule
- no substantial inference is required

If substantial inference is needed:

DO NOT create a Compliance Finding.

Create an Advisory Finding instead.


============================================================
BLOCKING FINDING FIELDS
============================================================

Every blocking finding must contain:

DIRECT_CONFLICT
Use:

YES

EVIDENCE
Exact quotation from Generated Content.

POLICY_SOURCE
Exactly one of:

BRAND INFORMATION
CAMPAIGN BRIEF
ADDITIONAL POLICY CONTEXT

POLICY_BASIS
Exact quotation from that source.

REASON
Explain the direct text-to-rule conflict.

REQUIRED_ACTION
Explain the minimum necessary correction.


============================================================
PART B — ADVISORY REVIEW
============================================================

Use ADVISORY for useful concerns that should not
be treated as hard blocking decisions.

Examples include:

- brand tone
- platform fit
- communication quality
- selling-point coverage
- unsupported product claims
- unsupported performance claims
- invented personal experience
- authenticity concerns
- wording that may create compliance risk
  but lacks enough supplied evidence
- inference-dependent medical or efficacy concerns
- possible sensitive wording
- approximate length issues
- hard-sell language
- any concern where the evidence-to-policy
  connection is uncertain

Do NOT force issues into predefined categories.

AREA is open-ended.


============================================================
PART C — DIAGNOSTIC SCORES
============================================================

Score:

BRAND_ALIGNMENT
1 = very poor
10 = excellent

TONE_MATCH
1 = very poor
10 = excellent

SELLING_POINT_COVERAGE
1 = very poor
10 = excellent

FACTUAL_CONSISTENCY
1 = very poor
10 = excellent

UNSUPPORTED_CLAIM_RISK
1 = very low risk
10 = very high risk

These scores are diagnostic signals only.

They do NOT determine:

- compliance
- pass/fail
- auto-revision

There is NO numerical acceptance threshold.


============================================================
OUTPUT FORMAT — VERY IMPORTANT
============================================================

DO NOT return JSON.

DO NOT return Markdown.

DO NOT use code fences.

Return simple KEY=VALUE lines only.

Every value must stay on ONE line.

Start with:

BEGIN_EVALUATION


Always output:

BRAND_ALIGNMENT=number
TONE_MATCH=number
SELLING_POINT_COVERAGE=number
FACTUAL_CONSISTENCY=number
UNSUPPORTED_CLAIM_RISK=number


Then:

COMPLIANCE_COUNT=number


For every blocking finding:

COMPLIANCE_1_DIRECT_CONFLICT=YES
COMPLIANCE_1_EVIDENCE=exact quotation from Generated Content
COMPLIANCE_1_POLICY_SOURCE=BRAND INFORMATION
COMPLIANCE_1_POLICY_BASIS=exact quotation from the stated source
COMPLIANCE_1_REASON=direct conflict explanation
COMPLIANCE_1_REQUIRED_ACTION=minimum necessary correction


Additional findings:

COMPLIANCE_2_DIRECT_CONFLICT=YES
COMPLIANCE_2_EVIDENCE=...
COMPLIANCE_2_POLICY_SOURCE=...
COMPLIANCE_2_POLICY_BASIS=...
COMPLIANCE_2_REASON=...
COMPLIANCE_2_REQUIRED_ACTION=...


If there are no valid direct blocking findings:

COMPLIANCE_COUNT=0


Then:

ADVISORY_COUNT=number


For advisory findings:

ADVISORY_1_AREA=open-ended review area
ADVISORY_1_EVIDENCE=relevant wording or blank
ADVISORY_1_REASON=why this deserves attention
ADVISORY_1_SUGGESTION=optional improvement


Continue numbering as needed.


If none:

ADVISORY_COUNT=0


Then:

REVIEW_NOTE_COUNT=number


Optional:

REVIEW_NOTE_1=note
REVIEW_NOTE_2=note


If none:

REVIEW_NOTE_COUNT=0


Finish with:

END_EVALUATION


============================================================
FINAL INSTRUCTION
============================================================

Judge ONLY the Generated Content.

A bad instruction in the Campaign Brief does not
make the final Generated Content bad if the model
did not follow that instruction.

When uncertain between BLOCKING and ADVISORY:

choose ADVISORY.

Blocking requires a direct,
source-grounded conflict.

Do not output commentary before BEGIN_EVALUATION.

Do not output commentary after END_EVALUATION.
"""


    response_text = generate_text(
        prompt=prompt,
        model_key=selected_judge,
        temperature=0.0,
    )


    evaluation = (
        parse_evaluation_response(
            response_text=response_text,
            generated_content=generated_content,
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            policy_context=policy_context,
        )
    )


    evaluation[
        "judge_model"
    ] = selected_judge


    return evaluation


# =========================================================
# Pairwise Response Parser
# =========================================================

def parse_pairwise_response(
    response_text: str,
) -> dict:
    """
    Parse simple pairwise KEY=VALUE output.
    """

    parsed = parse_key_value_response(
        response_text
    )


    if "PREFERENCE" not in parsed:

        preview = repr(
            clean_response_text(
                response_text
            )[:500]
        )

        raise ValueError(
            "Pairwise Judge response is missing "
            "PREFERENCE. "
            f"Raw response preview: {preview}"
        )


    preference = (
        parsed[
            "PREFERENCE"
        ]
        .strip()
        .lower()
    )


    if preference == "a":

        normalized_preference = "A"


    elif preference == "b":

        normalized_preference = "B"


    elif preference in {
        "tie",
        "equal",
        "same",
    }:

        normalized_preference = "tie"


    else:

        raise ValueError(
            "Pairwise Judge returned invalid "
            f"preference: {preference}"
        )


    return {
        "preference":
            normalized_preference,

        "reason":
            parsed.get(
                "REASON",
                "",
            ).strip(),

        "key_difference":
            parsed.get(
                "KEY_DIFFERENCE",
                "",
            ).strip(),
    }


# =========================================================
# Pairwise Evaluation
# =========================================================

def compare_contents_pairwise(
    brand_info: str,
    campaign_brief: str,
    content_a: str,
    content_b: str,
    policy_context: str = "",
    judge_model_key: str | None = None,
) -> dict:
    """
    Compare Content A and Content B without
    using an absolute numerical threshold.

    Pairwise evaluation remains separate from
    the blocking compliance decision.
    """

    selected_judge = (
        judge_model_key
        or JUDGE_MODEL_KEY
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
You are comparing two AI-generated
marketing contents.

Judge the actual contents.

Use ONLY the supplied information.

Do NOT invent:

- laws
- regulations
- platform policies
- product facts
- brand rules

from your own memory.


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
CONTENT A
============================================================

{content_a}


============================================================
CONTENT B
============================================================

{content_b}


============================================================
COMPARISON TASK
============================================================

Choose which content is better suited
for actual use.

Prioritize:

1. direct compliance with supplied rules
2. factual consistency
3. avoidance of unsupported claims
4. preservation of supplied product facts
5. brand alignment
6. campaign suitability
7. communication quality


IMPORTANT:

Judge the actual Content A and Content B.

A problematic request appearing in the Campaign Brief
does NOT make a content version non-compliant if that
content safely ignores the request.

A concern requiring substantial inference should
not be treated as a definitive policy violation.

However, unsupported or risky claims may still
reduce the content's overall suitability.

Do not prefer a version merely because it is:

- longer
- more creative
- more detailed
- more promotional


Possible decisions:

A
B
tie


============================================================
OUTPUT FORMAT
============================================================

DO NOT return JSON.

DO NOT use Markdown.

Return exactly:

BEGIN_PAIRWISE
PREFERENCE=A
REASON=brief evidence-based explanation
KEY_DIFFERENCE=main difference
END_PAIRWISE


PREFERENCE must be:

A
B
or
tie

Keep every value on one line.
"""


    response_text = generate_text(
        prompt=prompt,
        model_key=selected_judge,
        temperature=0.0,
    )


    result = parse_pairwise_response(
        response_text
    )


    result[
        "judge_model"
    ] = selected_judge


    return result