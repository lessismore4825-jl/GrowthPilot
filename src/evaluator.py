import re
import unicodedata

from src.llm_client import JUDGE_MODEL_KEY, generate_text


# =========================================================
# Constants
# =========================================================

SCORE_FIELDS = [
    "brand_alignment",
    "tone_match",
    "selling_point_coverage",
    "factual_consistency",
    "unsupported_claim_risk",
]

ALLOWED_POLICY_SOURCES = {
    "BRAND INFORMATION",
    "CAMPAIGN BRIEF",
    "ADDITIONAL POLICY CONTEXT",
}

VALID_CONTENT_ORIGINS = {
    "generated",
    "creator_draft",
    "unknown",
}

ALLOWED_REQUIREMENT_MATCH_MODES = {
    "EXACT",
    "SEMANTIC",
}

ALLOWED_ADVISORY_BASIS_TYPES = {
    "SUPPLIED_CONTEXT",
    "GENERAL_HEURISTIC",
    "SYSTEM_GROUNDING_REVIEW",
}


# =========================================================
# Basic Normalization
# =========================================================

def normalize_content_origin(value: str | None) -> str:
    """Normalize the origin of the submitted content."""

    normalized = str(value or "generated").strip().lower()

    aliases = {
        "ai": "generated",
        "ai_generated": "generated",
        "generated_content": "generated",
        "creator": "creator_draft",
        "creator_submitted": "creator_draft",
        "submitted": "creator_draft",
        "creator_draft": "creator_draft",
        "unknown": "unknown",
    }

    normalized = aliases.get(
        normalized,
        normalized,
    )

    if normalized not in VALID_CONTENT_ORIGINS:
        return "unknown"

    return normalized


def normalize_requirement_match_mode(
    value: str | None,
) -> str:
    """Normalize EXACT / SEMANTIC requirement matching mode."""

    normalized = str(
        value or "SEMANTIC"
    ).strip().upper()

    if normalized not in ALLOWED_REQUIREMENT_MATCH_MODES:
        return "SEMANTIC"

    return normalized


def normalize_requirements(
    requirements,
) -> list[dict]:
    """
    Normalize structured mandatory campaign requirements.

    Supported examples:

    ["BPA-free", "Portable design"]

    or:

    [
        {
            "requirement_id": "R1",
            "content": "#BrandCampaign",
            "match_mode": "EXACT",
        }
    ]

    Plain strings default to SEMANTIC.
    """

    normalized_requirements = []

    for index, raw_item in enumerate(
        requirements or [],
        start=1,
    ):
        if isinstance(
            raw_item,
            str,
        ):
            content = raw_item.strip()
            requirement_id = f"R{index}"
            match_mode = "SEMANTIC"

        elif isinstance(
            raw_item,
            dict,
        ):
            content = str(
                raw_item.get("content")
                or raw_item.get("requirement")
                or raw_item.get("item")
                or raw_item.get("missing_item")
                or ""
            ).strip()

            requirement_id = str(
                raw_item.get("requirement_id")
                or raw_item.get("id")
                or f"R{index}"
            ).strip()

            match_mode = normalize_requirement_match_mode(
                raw_item.get("match_mode")
            )

        else:
            continue

        if not content:
            continue

        if not requirement_id:
            requirement_id = f"R{index}"

        normalized_requirements.append(
            {
                "requirement_id": requirement_id,
                "content": content,
                "match_mode": match_mode,
            }
        )

    return normalized_requirements


def format_requirements_for_prompt(
    requirements: list[dict],
) -> str:
    """Render structured requirements for the Judge prompt."""

    if not requirements:
        return (
            "No structured mandatory campaign "
            "requirements were supplied."
        )

    return "\n".join(
        " | ".join(
            [
                f"ID={item['requirement_id']}",
                f"MATCH_MODE={item['match_mode']}",
                f"CONTENT={item['content']}",
            ]
        )
        for item in requirements
    )


def build_requirement_required_action(
    requirement: str,
    match_mode: str,
) -> str:
    """
    Build Requirement Required Action deterministically.

    The LLM decides whether a structured requirement is missing.
    Python decides the mandatory action so the Judge cannot introduce
    an unsupported marketing claim inside REQUIRED_ACTION.
    """

    requirement = str(
        requirement or ""
    ).strip()

    match_mode = normalize_requirement_match_mode(
        match_mode
    )

    if match_mode == "EXACT":
        return (
            f'Add the exact required content: "{requirement}". '
            "Make the smallest necessary change and do not "
            "introduce any unsupported claim, guarantee, "
            "creator experience, or additional product benefit."
        )

    return (
        f'Add a natural mention of the required concept: "{requirement}". '
        "Equivalent wording is acceptable. Make the smallest "
        "necessary change and do not introduce any unsupported "
        "claim, guarantee, creator experience, or additional "
        "product benefit."
    )


def normalize_advisory_basis_type(
    value: str | None,
) -> str:
    """Normalize Advisory provenance labels."""

    normalized = str(
        value or "GENERAL_HEURISTIC"
    ).strip().upper()

    if normalized not in ALLOWED_ADVISORY_BASIS_TYPES:
        return "GENERAL_HEURISTIC"

    return normalized


# =========================================================
# Response Parsing Helpers
# =========================================================

def clean_response_text(
    response_text: str,
) -> str:
    """Clean common LLM formatting artifacts."""

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

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    cleaned = re.sub(
        r"```(?:text|txt|json|python)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return (
        cleaned
        .replace("```", "")
        .strip()
    )


def parse_key_value_response(
    response_text: str,
) -> dict:
    """Parse simple line-based KEY=VALUE Judge output."""

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

        if value.endswith(","):
            value = value[:-1].rstrip()

        if len(value) >= 2:
            for left, right in [
                ('"', '"'),
                ("'", "'"),
                ("“", "”"),
                ("‘", "’"),
            ]:
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
            parsed[key] = value

    if not parsed:
        raise ValueError(
            "Judge response could not be parsed "
            "as key-value output. "
            f"Raw response preview: {cleaned[:500]!r}"
        )

    return parsed


def normalize_score(
    value,
    field_name: str,
):
    """Normalize a diagnostic score to 1-10."""

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    if isinstance(value, str):
        value = value.strip()

        match = re.match(
            r"^-?\d+(?:\.\d+)?",
            value,
        )

        if not match:
            raise ValueError(
                f"{field_name} must be numeric. "
                f"Received: {value}"
            )

        value = float(
            match.group()
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
            f"{field_name} must be between 1 and 10."
        )

    if float(value).is_integer():
        return int(value)

    return round(
        float(value),
        2,
    )


def normalize_policy_source(
    value: str,
) -> str:
    """Normalize common Policy Source variations."""

    normalized = re.sub(
        r"\s+",
        " ",
        str(value or "")
        .strip()
        .upper()
        .replace("_", " ")
        .replace("-", " "),
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


def get_indexed_numbers(
    parsed: dict,
    prefix: str,
) -> list:
    """Find indexes such as COMPLIANCE_1_* -> [1]."""

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
                    match.group(1)
                )
            )

    return sorted(
        indexes
    )


# =========================================================
# Deterministic Grounding Helpers
# =========================================================

def strip_wrapping_quotes(
    text: str,
) -> str:
    """Remove one matching outer quote pair."""

    value = str(
        text or ""
    ).strip()

    for left, right in [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
    ]:
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
    """Conservative normalization for provenance checks."""

    value = unicodedata.normalize(
        "NFKC",
        str(
            text or ""
        ),
    )

    value = (
        value
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

    return (
        re.sub(
            r"\s+",
            " ",
            value,
        )
        .strip()
        .casefold()
    )


def text_is_grounded(
    quoted_text: str,
    source_text: str,
) -> bool:
    """
    Verify a quotation actually appears in its claimed source.

    This is intentionally stricter than semantic similarity.
    """

    quote = strip_wrapping_quotes(
        quoted_text
    )

    source = str(
        source_text or ""
    )

    if not quote or not source:
        return False

    normalized_quote = normalize_for_grounding(
        quote
    )

    normalized_source = normalize_for_grounding(
        source
    )

    if not normalized_quote:
        return False

    if normalized_quote in normalized_source:
        return True

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

    return bool(
        compact_quote
        and compact_quote
        in compact_source
    )


def get_policy_source_text(
    policy_source: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
) -> str:
    """Return actual text belonging to a declared Policy Source."""

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
    """Normalize direct-conflict flags."""

    return (
        str(value or "")
        .strip()
        .upper()
        in {
            "YES",
            "Y",
            "TRUE",
            "1",
        }
    )


# =========================================================
# Compliance Findings
# =========================================================

def make_grounding_advisory(
    evidence: str,
    reason: str,
) -> dict:
    """Preserve a rejected Blocking proposal as a review signal."""

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
                "compliance finding because the required "
                "deterministic grounding could not be verified."
            ),

        "basis_type":
            "SYSTEM_GROUNDING_REVIEW",

        "basis_source":
            "",

        "basis_quote":
            "",
    }


def parse_compliance_findings(
    parsed: dict,
    generated_content: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
) -> tuple[
    list,
    list,
    list,
]:
    """Parse proposed Blocking Findings through deterministic gates."""

    findings = []
    review_notes = []
    downgraded_advisories = []

    for index in get_indexed_numbers(
        parsed,
        "COMPLIANCE",
    ):

        prefix = (
            f"COMPLIANCE_{index}"
        )

        direct_conflict = (
            parsed.get(
                f"{prefix}_DIRECT_CONFLICT",
                "",
            )
            .strip()
        )

        evidence = (
            parsed.get(
                f"{prefix}_EVIDENCE",
                "",
            )
            .strip()
        )

        policy_source = (
            normalize_policy_source(
                parsed.get(
                    f"{prefix}_POLICY_SOURCE",
                    "",
                )
            )
        )

        policy_basis = (
            parsed.get(
                f"{prefix}_POLICY_BASIS",
                "",
            )
            .strip()
        )

        reason = (
            parsed.get(
                f"{prefix}_REASON",
                "",
            )
            .strip()
        )

        required_action = (
            parsed.get(
                f"{prefix}_REQUIRED_ACTION",
                "",
            )
            .strip()
        )

        def downgrade(
            message: str,
        ):
            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_grounding_advisory(
                    evidence=evidence,
                    reason=message,
                )
            )

        if not (
            evidence
            and policy_source
            and policy_basis
            and reason
        ):
            downgrade(
                (
                    "Potential compliance finding "
                    f"{index} was not treated as blocking "
                    "because required grounding fields "
                    "were incomplete."
                )
            )

            continue

        if not is_yes(
            direct_conflict
        ):
            downgrade(
                (
                    "Potential compliance finding "
                    f"{index} was downgraded because "
                    "the Judge did not confirm a direct "
                    "text-to-rule conflict."
                )
            )

            continue

        if (
            policy_source
            not in ALLOWED_POLICY_SOURCES
        ):
            downgrade(
                (
                    "Potential compliance finding "
                    f"{index} referenced unsupported "
                    "policy source: "
                    f"{policy_source}. "
                    "It was not treated as blocking."
                )
            )

            continue

        if (
            policy_source
            == "ADDITIONAL POLICY CONTEXT"
            and not str(
                policy_context or ""
            ).strip()
        ):
            downgrade(
                (
                    "Potential compliance finding "
                    f"{index} cited ADDITIONAL POLICY "
                    "CONTEXT even though no additional "
                    "policy context was supplied."
                )
            )

            continue

        if not text_is_grounded(
            quoted_text=evidence,
            source_text=generated_content,
        ):
            downgrade(
                (
                    "Potential compliance finding "
                    f"{index} was downgraded because "
                    "its Evidence could not be verified "
                    "as an actual quotation from the "
                    "submitted content."
                )
            )

            continue

        source_text = (
            get_policy_source_text(
                policy_source=policy_source,
                brand_info=brand_info,
                campaign_brief=campaign_brief,
                policy_context=policy_context,
            )
        )

        if not text_is_grounded(
            quoted_text=policy_basis,
            source_text=source_text,
        ):
            downgrade(
                (
                    "Potential compliance finding "
                    f"{index} was downgraded because "
                    "its Policy Basis could not be verified "
                    "in the claimed Policy Source "
                    f"({policy_source})."
                )
            )

            continue

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
# Requirement Findings
# =========================================================

def make_requirement_review_advisory(
    requirement: str,
    reason: str,
) -> dict:
    """Preserve invalid Requirement proposals as review signals."""

    return {
        "area":
            "Requirement Grounding Review",

        "evidence":
            requirement,

        "reason":
            reason,

        "suggestion":
            (
                "Review this requirement manually. "
                "It was not treated as a mandatory "
                "Requirement Finding because its "
                "structured grounding could not be verified."
            ),

        "basis_type":
            "SYSTEM_GROUNDING_REVIEW",

        "basis_source":
            "",

        "basis_quote":
            "",
    }


def parse_requirement_findings(
    parsed: dict,
    requirements: list[dict],
    submitted_content: str,
) -> tuple[
    list,
    list,
    list,
]:
    """
    Parse mandatory Campaign Requirement omissions.

    EXACT absence is deterministic.

    SEMANTIC absence is a Judge semantic decision and should
    receive Cross-Judge confirmation before automatic action.
    """

    findings = []
    review_notes = []
    downgraded_advisories = []

    requirement_map = {
        str(
            item[
                "requirement_id"
            ]
        ):
            item

        for item
        in requirements
    }

    for index in get_indexed_numbers(
        parsed,
        "REQUIREMENT",
    ):

        prefix = (
            f"REQUIREMENT_{index}"
        )

        requirement_id = str(
            parsed.get(
                f"{prefix}_REQUIREMENT_ID",
                "",
            )
        ).strip()

        reason = str(
            parsed.get(
                f"{prefix}_REASON",
                "",
            )
        ).strip()

        if (
            not requirement_id
            or requirement_id
            not in requirement_map
        ):
            message = (
                "Potential requirement finding "
                f"{index} referenced an unknown "
                "structured requirement ID: "
                f"{requirement_id or '<empty>'}."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_requirement_review_advisory(
                    requirement=requirement_id,
                    reason=message,
                )
            )

            continue

        requirement = (
            requirement_map[
                requirement_id
            ]
        )

        content = (
            requirement[
                "content"
            ]
        )

        match_mode = (
            requirement[
                "match_mode"
            ]
        )

        if not reason:

            message = (
                "Potential requirement finding "
                f"{index} for "
                f"{requirement_id} was rejected "
                "because REASON was empty."
            )

            review_notes.append(
                message
            )

            downgraded_advisories.append(
                make_requirement_review_advisory(
                    requirement=content,
                    reason=message,
                )
            )

            continue

        if (
            match_mode
            == "EXACT"
        ):

            if text_is_grounded(
                quoted_text=content,
                source_text=submitted_content,
            ):
                review_notes.append(
                    (
                        "Potential requirement finding "
                        f"{index} for "
                        f"{requirement_id} was rejected "
                        "because the EXACT required "
                        "content is already present."
                    )
                )

                continue

            verification_mode = (
                "DETERMINISTIC_EXACT_ABSENCE"
            )

        else:

            verification_mode = (
                "JUDGE_SEMANTIC_ABSENCE"
            )

        findings.append(
            {
                "requirement_id":
                    requirement_id,

                "requirement":
                    content,

                "match_mode":
                    match_mode,

                "reason":
                    reason,

                "required_action":
                    build_requirement_required_action(
                        requirement=content,
                        match_mode=match_mode,
                    ),

                "verification_mode":
                    verification_mode,
            }
        )

    return (
        findings,
        review_notes,
        downgraded_advisories,
    )


# =========================================================
# Advisory Findings + Provenance
# =========================================================

def parse_advisory_findings(
    parsed: dict,
) -> list:
    """Parse open-ended non-blocking Advisory Findings."""

    findings = []

    for index in get_indexed_numbers(
        parsed,
        "ADVISORY",
    ):

        prefix = (
            f"ADVISORY_{index}"
        )

        area = (
            parsed.get(
                f"{prefix}_AREA",
                "General",
            )
            .strip()
        )

        evidence = (
            parsed.get(
                f"{prefix}_EVIDENCE",
                "",
            )
            .strip()
        )

        reason = (
            parsed.get(
                f"{prefix}_REASON",
                "",
            )
            .strip()
        )

        suggestion = (
            parsed.get(
                f"{prefix}_SUGGESTION",
                "",
            )
            .strip()
        )

        basis_type = (
            normalize_advisory_basis_type(
                parsed.get(
                    f"{prefix}_BASIS_TYPE",
                    "GENERAL_HEURISTIC",
                )
            )
        )

        basis_source = (
            normalize_policy_source(
                parsed.get(
                    f"{prefix}_BASIS_SOURCE",
                    "",
                )
            )
        )

        basis_quote = (
            parsed.get(
                f"{prefix}_BASIS_QUOTE",
                "",
            )
            .strip()
        )

        if not reason:
            continue

        findings.append(
            {
                "area":
                    area
                    or "General",

                "evidence":
                    evidence,

                "reason":
                    reason,

                "suggestion":
                    suggestion,

                "basis_type":
                    basis_type,

                "basis_source":
                    basis_source,

                "basis_quote":
                    basis_quote,
            }
        )

    return findings


def validate_advisory_provenance(
    findings: list,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
) -> tuple[
    list,
    list,
]:
    """
    Validate Advisory source provenance.

    If SUPPLIED_CONTEXT cannot be verified, the advice stays
    visible but is relabeled GENERAL_HEURISTIC.

    This deterministic check validates quotation provenance.

    The prompt separately instructs the Judge that the quote
    must also directly support the Advisory conclusion.
    """

    validated = []
    review_notes = []

    for index, finding in enumerate(
        findings,
        start=1,
    ):

        item = dict(
            finding
        )

        basis_type = (
            item.get(
                "basis_type",
                "GENERAL_HEURISTIC",
            )
        )

        if (
            basis_type
            == "SYSTEM_GROUNDING_REVIEW"
        ):
            validated.append(
                item
            )

            continue

        if (
            basis_type
            == "SUPPLIED_CONTEXT"
        ):

            basis_source = (
                normalize_policy_source(
                    item.get(
                        "basis_source",
                        "",
                    )
                )
            )

            basis_quote = str(
                item.get(
                    "basis_quote",
                    "",
                )
                or ""
            ).strip()

            source_text = (
                get_policy_source_text(
                    policy_source=basis_source,
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    policy_context=policy_context,
                )
            )

            valid_provenance = (
                basis_source
                in ALLOWED_POLICY_SOURCES

                and bool(
                    basis_quote
                )

                and text_is_grounded(
                    quoted_text=basis_quote,
                    source_text=source_text,
                )
            )

            if valid_provenance:

                item[
                    "basis_source"
                ] = (
                    basis_source
                )

                item[
                    "basis_quote"
                ] = (
                    strip_wrapping_quotes(
                        basis_quote
                    )
                )

            else:

                review_notes.append(
                    (
                        "Advisory finding "
                        f"{index} claimed "
                        "SUPPLIED_CONTEXT but its "
                        "basis provenance could not "
                        "be verified. It was relabeled "
                        "GENERAL_HEURISTIC."
                    )
                )

                item[
                    "basis_type"
                ] = (
                    "GENERAL_HEURISTIC"
                )

                item[
                    "basis_source"
                ] = ""

                item[
                    "basis_quote"
                ] = ""

        else:

            item[
                "basis_type"
            ] = (
                "GENERAL_HEURISTIC"
            )

            item[
                "basis_source"
            ] = ""

            item[
                "basis_quote"
            ] = ""

        validated.append(
            item
        )

    return (
        validated,
        review_notes,
    )


# =========================================================
# Review Notes + Diagnostic Score
# =========================================================

def parse_review_notes(
    parsed: dict,
) -> list:
    """Parse REVIEW_NOTE_1=... style notes."""

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
        key=lambda item:
            item[0]
    )

    return [
        note
        for _, note
        in indexed_notes
    ]


def calculate_heuristic_score(
    evaluation: dict,
) -> float:
    """
    Comparative diagnostic signal only.

    This is NOT a calibrated acceptance threshold.
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
        ]
        * 0.20

        + evaluation[
            "tone_match"
        ]
        * 0.15

        + evaluation[
            "selling_point_coverage"
        ]
        * 0.20

        + evaluation[
            "factual_consistency"
        ]
        * 0.30

        + claim_safety
        * 0.15
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
    requirements=None,
    content_origin: str = "generated",
) -> dict:
    """Parse GrowthPilot Evaluation Architecture v2.2 output."""

    parsed = (
        parse_key_value_response(
            response_text
        )
    )

    normalized_requirements = (
        normalize_requirements(
            requirements
        )
    )

    normalized_origin = (
        normalize_content_origin(
            content_origin
        )
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

        if (
            response_key
            not in parsed
        ):
            raise ValueError(
                "Judge response is missing "
                f"{response_key}. "
                "Raw response preview: "
                f"{clean_response_text(response_text)[:500]!r}"
            )

        evaluation[
            field_name
        ] = (
            normalize_score(
                parsed[
                    response_key
                ],
                field_name,
            )
        )

    (
        compliance_findings,
        compliance_review_notes,
        compliance_downgrades,
    ) = (
        parse_compliance_findings(
            parsed=parsed,
            generated_content=generated_content,
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            policy_context=policy_context,
        )
    )

    (
        requirement_findings,
        requirement_review_notes,
        requirement_downgrades,
    ) = (
        parse_requirement_findings(
            parsed=parsed,
            requirements=normalized_requirements,
            submitted_content=generated_content,
        )
    )

    model_advisories = (
        parse_advisory_findings(
            parsed
        )
    )

    all_advisories = (
        model_advisories
        + compliance_downgrades
        + requirement_downgrades
    )

    (
        validated_advisories,
        advisory_review_notes,
    ) = (
        validate_advisory_provenance(
            findings=all_advisories,
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            policy_context=policy_context,
        )
    )

    model_review_notes = (
        parse_review_notes(
            parsed
        )
    )

    evaluation[
        "compliance_findings"
    ] = (
        compliance_findings
    )

    evaluation[
        "requirement_findings"
    ] = (
        requirement_findings
    )

    evaluation[
        "advisory_findings"
    ] = (
        validated_advisories
    )

    evaluation[
        "review_notes"
    ] = (
        model_review_notes
        + compliance_review_notes
        + requirement_review_notes
        + advisory_review_notes
    )

    evaluation[
        "heuristic_composite_score"
    ] = (
        calculate_heuristic_score(
            evaluation
        )
    )

    blocking_count = len(
        compliance_findings
    )

    requirement_count = len(
        requirement_findings
    )

    evaluation[
        "blocking_compliance_issue_count"
    ] = (
        blocking_count
    )

    evaluation[
        "compliance_status"
    ] = (
        "BLOCKING_ISSUES_DETECTED"

        if blocking_count > 0

        else
        "NO_BLOCKING_ISSUES_DETECTED"
    )

    evaluation[
        "requirement_finding_count"
    ] = (
        requirement_count
    )

    # Compatibility / experiment alias.
    evaluation[
        "requirement_missing_count"
    ] = (
        requirement_count
    )

    evaluation[
        "requirement_status"
    ] = (
        "REQUIREMENTS_MISSING"

        if requirement_count > 0

        else
        "NO_REQUIREMENTS_MISSING"
    )

    evaluation[
        "mandatory_action_count"
    ] = (
        blocking_count
        + requirement_count
    )

    if (
        blocking_count > 0
        and requirement_count > 0
    ):

        mandatory_action_status = (
            "COMPLIANCE_AND_REQUIREMENT_ACTION"
        )

    elif blocking_count > 0:

        mandatory_action_status = (
            "COMPLIANCE_ACTION"
        )

    elif requirement_count > 0:

        mandatory_action_status = (
            "REQUIREMENT_ACTION"
        )

    else:

        mandatory_action_status = (
            "NO_MANDATORY_ACTION"
        )

    evaluation[
        "mandatory_action_status"
    ] = (
        mandatory_action_status
    )

    evaluation[
        "content_origin"
    ] = (
        normalized_origin
    )

    evaluation[
        "structured_requirement_count"
    ] = (
        len(
            normalized_requirements
        )
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
    requirements=None,
    content_origin: str = "generated",
) -> dict:
    """
    GrowthPilot Evaluation Architecture v2.2.

    Existing callers remain valid because the v2 parameters
    are optional.
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

    normalized_requirements = (
        normalize_requirements(
            requirements
        )
    )

    requirements_display = (
        format_requirements_for_prompt(
            normalized_requirements
        )
    )

    normalized_origin = (
        normalize_content_origin(
            content_origin
        )
    )

    if (
        normalized_origin
        == "creator_draft"
    ):

        origin_instruction = """
The submitted content is CREATOR-AUTHORED / CREATOR-SUBMITTED.

Do NOT assume a first-person experience, opinion or endorsement is fabricated
merely because Brand Information does not independently prove that the Creator
experienced it.

Examples that are NOT automatically authenticity violations:

- "我最近在用..."
- "I really like this workflow"
- "Honestly kinda obsessed with this workflow rn!!!"

Still review factual product claims, efficacy claims, guarantees, statistics
and technical specifications normally.
""".strip()

    elif (
        normalized_origin
        == "generated"
    ):

        origin_instruction = """
The submitted content may be AI-generated.

Unsupported factual claims or newly invented first-person experiences may be
Advisory concerns when appropriate, but only direct supplied-rule conflicts
may be Blocking.
""".strip()

    else:

        origin_instruction = """
The content origin is unknown.

Do not assume first-person experience is fabricated solely because it is absent
from Brand Information. Evaluate factual and policy claims from the supplied
sources.
""".strip()

    prompt = f"""
You are a strict AI marketing content reviewer for GrowthPilot.

Evaluate ONLY the actual SUBMITTED CONTENT.

Separate four layers:

1. BLOCKING COMPLIANCE FINDINGS
2. MANDATORY REQUIREMENT FINDINGS
3. NON-BLOCKING ADVISORY FINDINGS
4. DIAGNOSTIC SCORES

There is NO numerical acceptance threshold.


============================================================
CONTENT ORIGIN
============================================================

{normalized_origin}

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

{policy_context_display}


============================================================
STRUCTURED MANDATORY CAMPAIGN REQUIREMENTS
============================================================

{requirements_display}

These structured requirements are the ONLY items that may become formal
Requirement Findings.

Do NOT infer a mandatory requirement merely because a product fact,
selling point, target audience, platform preference or general creative
idea appears elsewhere in the materials.


============================================================
SUBMITTED CONTENT
============================================================

{generated_content}


============================================================
CORE PRODUCT PRINCIPLE
============================================================

Policy decides what must be corrected.
Campaign requirements decide what must be completed.
AI advises what could be improved.
Human decides what should ultimately be used.

Judge the SUBMITTED CONTENT itself.

A problematic instruction in Campaign Brief does NOT automatically make
Submitted Content non-compliant.

If Campaign Brief requests something prohibited but Submitted Content
safely ignores it, DO NOT create a Blocking Finding.

If a structured requirement conflicts with a higher-priority Brand or
supplied Policy restriction, do NOT require the unsafe item.

Mention the conflict in REVIEW_NOTE instead.


============================================================
PART A — BLOCKING COMPLIANCE REVIEW
============================================================

A Blocking Finding is allowed ONLY when there is a DIRECT conflict between:

A specific phrase that ACTUALLY APPEARS in SUBMITTED CONTENT

AND

an explicit fact, restriction or rule that ACTUALLY APPEARS in one of:

- BRAND INFORMATION
- CAMPAIGN BRIEF
- ADDITIONAL POLICY CONTEXT

Every Blocking Finding must satisfy ALL conditions:

1. EVIDENCE is an EXACT quotation from SUBMITTED CONTENT.
2. POLICY_BASIS is an EXACT quotation from the declared Policy Source.
3. POLICY_SOURCE is the source containing the violated rule or fact.
4. The conflict is direct.
5. No substantial inference is required.
6. DIRECT_CONFLICT=YES.

UNSUPPORTED does NOT automatically mean Blocking.

If a concern requires interpretation, uncertainty or inference, use
ADVISORY.

POLICY_SOURCE means the source containing the rule or fact actually
violated.

It does not mean the source that motivated the claim.

Do NOT paraphrase EVIDENCE or POLICY_BASIS.


============================================================
PART B — MANDATORY REQUIREMENT COVERAGE
============================================================

Requirement Findings are NOT Compliance Findings.

Use REQUIREMENT only when a structured mandatory Campaign Requirement
listed above is genuinely missing from Submitted Content.

For MATCH_MODE=EXACT:

- the required content must appear literally.

For MATCH_MODE=SEMANTIC:

- equivalent meaning is sufficient;
- do NOT mark it missing merely because exact words are absent.

Example:

Requirement:
Portable design

MATCH_MODE=SEMANTIC

Submitted Content:
"放包里基本没什么负担"

Result:

Portability is already communicated.

Do NOT mark it missing just because "Portable design" is not written
literally.

An omission has no problematic Evidence quotation.

Therefore do NOT invent fake Evidence for Requirement Findings.

Ground the finding using the structured REQUIREMENT_ID.

Create a Requirement Finding only when:

1. REQUIREMENT_ID exists in the structured list;
2. the requirement is genuinely missing according to MATCH_MODE;
3. the requirement does not conflict with higher-priority supplied rules.

Do NOT turn an omission into Blocking Compliance.

IMPORTANT:

Do NOT write a free-form replacement marketing claim for Requirement
Required Action.

Only output REQUIREMENT_ID and REASON.

Python will create the mandatory action deterministically from the
structured Requirement itself.


============================================================
PART C — ADVISORY REVIEW
============================================================

Use ADVISORY for useful non-blocking concerns such as:

- brand tone
- creator fit
- platform fit
- communication quality
- optional selling-point coverage
- unsupported but not directly prohibited claims
- uncertain risk
- approximate length
- hard-sell language
- wording quality

Do NOT force concerns into a closed taxonomy.

AREA is open-ended.

When CONTENT ORIGIN is creator_draft:

Do NOT call first-person language fabricated merely because supplied
materials do not prove the Creator's personal experience.

The first-person form itself is not an authenticity violation.

You may still review a factual, efficacy, guarantee, safety or technical
claim inside that first-person statement.


============================================================
ADVISORY PROVENANCE
============================================================

Every Advisory must declare BASIS_TYPE as one of:

SUPPLIED_CONTEXT
GENERAL_HEURISTIC

Use SUPPLIED_CONTEXT ONLY when an exact supplied quote DIRECTLY SUPPORTS
the actual Advisory conclusion.

This is stricter than topical relevance.

A source merely mentioning the same brand, product, platform, audience or
topic is NOT enough.

Example:

Source:

"AeroBottle is an active-lifestyle drinkware brand."

This does NOT by itself support:

"The Creator must explicitly name AeroBottle in the caption."

That recommendation relies on a general marketing convention, so use:

BASIS_TYPE=GENERAL_HEURISTIC


Example:

Source:

"Its communication should be clear, technical, and useful without
exaggeration."

Submitted Content uses strongly promotional hype language.

This quote directly supports a Tone Advisory, so use:

BASIS_TYPE=SUPPLIED_CONTEXT


When BASIS_TYPE=SUPPLIED_CONTEXT:

- BASIS_SOURCE must be one allowed supplied source;
- BASIS_QUOTE must be an exact quotation from that source;
- the quote must directly support the Advisory conclusion.

Use GENERAL_HEURISTIC when advice comes from general marketing, platform,
copywriting or model knowledge rather than an explicit supplied statement.

Do NOT present a general heuristic as supplied policy.


============================================================
ADVISORY SUGGESTION SAFETY — v2.2
============================================================

ADVISORY_SUGGESTION must be an EDIT ACTION only.

It must describe WHAT TO CHANGE, not WRITE THE NEW MARKETING COPY.

Do NOT provide:

- example replacement sentences
- example captions
- example hooks
- quoted rewrite text
- newly composed promotional phrases
- new numerical claims
- new product benefits
- new creator experiences
- stronger promises

Do NOT use phrases such as:

- "for example: ..."
- "e.g. ..."
- "say: ..."
- "write: ..."
- "replace with: ..."
- "add: <new promotional sentence>"

unless the requested addition is a literal supplied EXACT requirement.

For normal Advisory Findings, prefer short action guidance such as:

- Reduce casual slang and emojis to better match the supplied professional tone.
- Make the specification-focused structure clearer using only verified product facts already supplied.
- Improve readability with shorter sentences or clearer structure.
- Remove or soften the unsupported outcome claim without replacing it with a new benefit.
- Consider mentioning an optional verified selling point only if it fits naturally.

The suggestion itself must NOT become a second content-generation channel.

A good Advisory identifies the issue and gives a bounded editing direction.

It does NOT draft the user's final marketing sentence.

Do NOT introduce or recommend inventing:

- new product facts
- new product functions
- guarantees
- unsupported efficacy claims
- unsupported performance outcomes
- medical or health outcomes
- certifications or research
- new Creator traits
- new Creator personal experiences
- unsupported usage duration
- unsupported battery or performance guarantees

If a safe action cannot be stated without inventing content, use conservative
action guidance only.


============================================================
PART D — DIAGNOSTIC SCORES
============================================================

Score each from 1 to 10:

BRAND_ALIGNMENT
TONE_MATCH
SELLING_POINT_COVERAGE
FACTUAL_CONSISTENCY
UNSUPPORTED_CLAIM_RISK

For the first four:

1 = very poor
10 = excellent

For UNSUPPORTED_CLAIM_RISK:

1 = very low risk
10 = very high risk

These scores are descriptive signals only.

They do NOT determine:

- compliance
- requirement completion
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


For every Blocking Finding:

COMPLIANCE_1_DIRECT_CONFLICT=YES
COMPLIANCE_1_EVIDENCE=exact quotation from Submitted Content
COMPLIANCE_1_POLICY_SOURCE=BRAND INFORMATION
COMPLIANCE_1_POLICY_BASIS=exact quotation from stated source
COMPLIANCE_1_REASON=direct conflict explanation
COMPLIANCE_1_REQUIRED_ACTION=minimum necessary correction


Continue numbering when needed.

If none:

COMPLIANCE_COUNT=0


Then:

REQUIREMENT_COUNT=number


For every missing structured Requirement:

REQUIREMENT_1_REQUIREMENT_ID=R1
REQUIREMENT_1_REASON=why the structured requirement is missing


Do NOT output REQUIREMENT REQUIRED_ACTION.

Python generates it deterministically.


If none:

REQUIREMENT_COUNT=0


Then:

ADVISORY_COUNT=number


For every Advisory:

ADVISORY_1_AREA=open-ended review area
ADVISORY_1_EVIDENCE=relevant Submitted Content wording or blank
ADVISORY_1_REASON=why this deserves attention
ADVISORY_1_SUGGESTION=short edit action only; no example rewrite or new marketing sentence
ADVISORY_1_BASIS_TYPE=SUPPLIED_CONTEXT or GENERAL_HEURISTIC
ADVISORY_1_BASIS_SOURCE=BRAND INFORMATION or CAMPAIGN BRIEF or ADDITIONAL POLICY CONTEXT or blank
ADVISORY_1_BASIS_QUOTE=exact supplied-source quotation or blank


For GENERAL_HEURISTIC use blank BASIS_SOURCE and BASIS_QUOTE.

For ADVISORY_SUGGESTION:

- give an action, not replacement copy;
- do not include example wording;
- do not introduce a new fact or claim.


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

Judge ONLY Submitted Content.

A bad Campaign Brief instruction does not make final content bad when the
content safely ignores it.

A missing structured Requirement is a REQUIREMENT FINDING, not Blocking
Compliance.

When uncertain between BLOCKING and ADVISORY, choose ADVISORY.

Blocking requires a direct, source-grounded conflict.

ADVISORY_SUGGESTION must be bounded edit guidance only.

Never use it to generate new campaign copy, example rewrite sentences,
unsupported numbers or new claims.

Do not output commentary before BEGIN_EVALUATION.

Do not output commentary after END_EVALUATION.
"""

    response_text = (
        generate_text(
            prompt=prompt,
            model_key=selected_judge,
            temperature=0.0,
        )
    )

    evaluation = (
        parse_evaluation_response(
            response_text=response_text,
            generated_content=generated_content,
            brand_info=brand_info,
            campaign_brief=campaign_brief,
            policy_context=policy_context,
            requirements=normalized_requirements,
            content_origin=normalized_origin,
        )
    )

    evaluation[
        "judge_model"
    ] = (
        selected_judge
    )

    return evaluation


# =========================================================
# Pairwise Evaluation
# =========================================================

def parse_pairwise_response(
    response_text: str,
) -> dict:
    """Parse simple pairwise KEY=VALUE output."""

    parsed = (
        parse_key_value_response(
            response_text
        )
    )

    if (
        "PREFERENCE"
        not in parsed
    ):
        raise ValueError(
            "Pairwise Judge response is missing "
            "PREFERENCE. "
            "Raw response preview: "
            f"{clean_response_text(response_text)[:500]!r}"
        )

    preference = (
        parsed[
            "PREFERENCE"
        ]
        .strip()
        .lower()
    )

    if (
        preference
        == "a"
    ):

        normalized_preference = (
            "A"
        )

    elif (
        preference
        == "b"
    ):

        normalized_preference = (
            "B"
        )

    elif preference in {
        "tie",
        "equal",
        "same",
    }:

        normalized_preference = (
            "tie"
        )

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


def compare_contents_pairwise(
    brand_info: str,
    campaign_brief: str,
    content_a: str,
    content_b: str,
    policy_context: str = "",
    judge_model_key: str | None = None,
) -> dict:
    """
    Compare two content versions without using a numerical
    pass/fail threshold.

    Kept backward-compatible with Benchmark A.
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
You are comparing two marketing content versions.

Judge the actual contents.

Use ONLY supplied information.

Do NOT invent:

- laws
- regulations
- platform policies
- product facts
- brand rules


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

Choose which content is better suited for actual use.

Prioritize:

1. direct compliance with supplied rules
2. factual consistency
3. avoidance of unsupported claims
4. preservation of supplied product facts
5. brand alignment
6. campaign suitability
7. communication quality

A problematic request in Campaign Brief does NOT make a version
non-compliant if that version safely ignores the request.

A concern requiring substantial inference should not be treated as a
definitive policy violation.

Unsupported or risky claims may still reduce overall suitability.

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

    response_text = (
        generate_text(
            prompt=prompt,
            model_key=selected_judge,
            temperature=0.0,
        )
    )

    result = (
        parse_pairwise_response(
            response_text
        )
    )

    result[
        "judge_model"
    ] = (
        selected_judge
    )

    return result