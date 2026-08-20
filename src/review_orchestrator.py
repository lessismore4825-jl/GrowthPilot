from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.evaluator import evaluate_content


# =========================================================
# Review Modes
# =========================================================

FAST_REVIEW = "fast"
CROSS_JUDGE_REVIEW = "cross_judge"


# =========================================================
# Default Judge Configuration
# =========================================================

DEFAULT_FAST_JUDGE = "step"

DEFAULT_CROSS_JUDGES = (
    "step",
    "qwen",
)


# =========================================================
# Final Routing Status
# =========================================================

NO_MANDATORY_ACTION = "NO_MANDATORY_ACTION"

COMPLIANCE_ACTION = "COMPLIANCE_ACTION"

REQUIREMENT_ACTION = "REQUIREMENT_ACTION"

COMPLIANCE_AND_REQUIREMENT_ACTION = (
    "COMPLIANCE_AND_REQUIREMENT_ACTION"
)

HUMAN_REVIEW_REQUIRED = (
    "HUMAN_REVIEW_REQUIRED"
)

REVIEW_ERROR = "REVIEW_ERROR"


# =========================================================
# Diagnostic Score Fields
# =========================================================

DIAGNOSTIC_FIELDS = [
    "brand_alignment",
    "tone_match",
    "selling_point_coverage",
    "factual_consistency",
    "unsupported_claim_risk",
    "heuristic_composite_score",
]


# =========================================================
# Text Normalization
# =========================================================

def normalize_match_text(
    value: Any,
) -> str:
    """
    Normalize text for conservative deterministic matching.

    This function is used only for cross-judge comparison.

    It does NOT perform semantic similarity matching.

    We intentionally avoid fuzzy semantic thresholds here because
    cross-judge consensus is used to authorize mandatory product
    actions. When deterministic evidence overlap cannot be proven,
    the safer behavior is Human Review rather than guessing that
    two findings are equivalent.
    """

    text = str(
        value or ""
    ).strip().lower()

    return "".join(
        char
        for char in text
        if char.isalnum()
    )


# =========================================================
# Compliance Finding Matching
# =========================================================

def compliance_evidence_matches(
    left_finding: dict,
    right_finding: dict,
) -> bool:
    """
    Determine whether two compliance findings refer to the same
    submitted-content evidence.

    The evaluator architecture requires compliance findings to carry
    direct draft evidence. Therefore evidence overlap is the safest
    deterministic cross-judge matching signal.

    Matching is intentionally conservative:

    - normalized evidence is equal, OR
    - one normalized evidence span contains the other.

    No model-based semantic matching is performed here.
    """

    left_evidence = normalize_match_text(
        left_finding.get(
            "evidence",
            "",
        )
    )

    right_evidence = normalize_match_text(
        right_finding.get(
            "evidence",
            "",
        )
    )

    if (
        not left_evidence
        or not right_evidence
    ):
        return False

    if (
        left_evidence
        == right_evidence
    ):
        return True

    if (
        left_evidence
        in right_evidence
    ):
        return True

    if (
        right_evidence
        in left_evidence
    ):
        return True

    return False


def match_compliance_findings(
    first_findings: list[dict],
    second_findings: list[dict],
    first_judge: str,
    second_judge: str,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:
    """
    Match compliance findings across two judges.

    Returns:

    1. consensus findings
    2. findings reported only by first judge
    3. findings reported only by second judge

    Only consensus findings are eligible for automatic mandatory
    revision in Cross-Judge Review mode.
    """

    consensus_findings = []

    unmatched_second_indexes = set(
        range(
            len(
                second_findings
            )
        )
    )

    unmatched_first = []

    for first_finding in first_findings:

        matched_index = None

        for second_index in list(
            unmatched_second_indexes
        ):

            second_finding = (
                second_findings[
                    second_index
                ]
            )

            if compliance_evidence_matches(
                first_finding,
                second_finding,
            ):
                matched_index = (
                    second_index
                )

                break

        if matched_index is None:

            unmatched_first.append(
                deepcopy(
                    first_finding
                )
            )

            continue

        second_finding = (
            second_findings[
                matched_index
            ]
        )

        unmatched_second_indexes.remove(
            matched_index
        )

        merged = deepcopy(
            first_finding
        )

        merged[
            "reported_by"
        ] = [
            first_judge,
            second_judge,
        ]

        merged[
            "cross_judge_match"
        ] = "EVIDENCE_OVERLAP"

        merged[
            "judge_evidence"
        ] = {
            first_judge:
                first_finding.get(
                    "evidence",
                    "",
                ),

            second_judge:
                second_finding.get(
                    "evidence",
                    "",
                ),
        }

        merged[
            "judge_reasons"
        ] = {
            first_judge:
                first_finding.get(
                    "reason",
                    "",
                ),

            second_judge:
                second_finding.get(
                    "reason",
                    "",
                ),
        }

        consensus_findings.append(
            merged
        )

    unmatched_second = [
        deepcopy(
            second_findings[
                index
            ]
        )
        for index
        in sorted(
            unmatched_second_indexes
        )
    ]

    for finding in unmatched_first:

        finding[
            "reported_by"
        ] = [
            first_judge
        ]

    for finding in unmatched_second:

        finding[
            "reported_by"
        ] = [
            second_judge
        ]

    return (
        consensus_findings,
        unmatched_first,
        unmatched_second,
    )


# =========================================================
# Requirement Finding Matching
# =========================================================

def requirement_match_key(
    finding: dict,
) -> tuple[str, str]:
    """
    Build a deterministic key for structured requirement matching.

    Requirement ID is the primary key.

    Requirement text is used as a fallback when the ID is missing.
    """

    requirement_id = str(
        finding.get(
            "requirement_id",
            "",
        )
        or ""
    ).strip()

    requirement_text = normalize_match_text(
        finding.get(
            "requirement",
            finding.get(
                "content",
                "",
            ),
        )
    )

    return (
        requirement_id,
        requirement_text,
    )


def requirement_findings_match(
    first_finding: dict,
    second_finding: dict,
) -> bool:
    """
    Determine whether two requirement findings refer to the same
    structured Must Mention item.
    """

    first_id, first_text = (
        requirement_match_key(
            first_finding
        )
    )

    second_id, second_text = (
        requirement_match_key(
            second_finding
        )
    )

    if (
        first_id
        and second_id
    ):
        return (
            first_id
            == second_id
        )

    if (
        first_text
        and second_text
    ):
        return (
            first_text
            == second_text
        )

    return False


def match_requirement_findings(
    first_findings: list[dict],
    second_findings: list[dict],
    first_judge: str,
    second_judge: str,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:
    """
    Match structured requirement findings across two judges.

    Only the same requirement independently identified as missing by
    both judges becomes a Cross-Judge mandatory requirement action.
    """

    consensus_findings = []

    unmatched_second_indexes = set(
        range(
            len(
                second_findings
            )
        )
    )

    unmatched_first = []

    for first_finding in first_findings:

        matched_index = None

        for second_index in list(
            unmatched_second_indexes
        ):

            second_finding = (
                second_findings[
                    second_index
                ]
            )

            if requirement_findings_match(
                first_finding,
                second_finding,
            ):
                matched_index = (
                    second_index
                )

                break

        if matched_index is None:

            unmatched_first.append(
                deepcopy(
                    first_finding
                )
            )

            continue

        second_finding = (
            second_findings[
                matched_index
            ]
        )

        unmatched_second_indexes.remove(
            matched_index
        )

        merged = deepcopy(
            first_finding
        )

        merged[
            "reported_by"
        ] = [
            first_judge,
            second_judge,
        ]

        merged[
            "cross_judge_match"
        ] = (
            "STRUCTURED_REQUIREMENT_ID"
        )

        merged[
            "judge_reasons"
        ] = {
            first_judge:
                first_finding.get(
                    "reason",
                    "",
                ),

            second_judge:
                second_finding.get(
                    "reason",
                    "",
                ),
        }

        consensus_findings.append(
            merged
        )

    unmatched_second = [
        deepcopy(
            second_findings[
                index
            ]
        )
        for index
        in sorted(
            unmatched_second_indexes
        )
    ]

    for finding in unmatched_first:

        finding[
            "reported_by"
        ] = [
            first_judge
        ]

    for finding in unmatched_second:

        finding[
            "reported_by"
        ] = [
            second_judge
        ]

    return (
        consensus_findings,
        unmatched_first,
        unmatched_second,
    )


# =========================================================
# Advisory Merge
# =========================================================

def advisory_identity(
    finding: dict,
) -> tuple:
    """
    Build a deterministic identity for optional advisory findings.

    Advisory findings do not control mandatory routing.

    Therefore exact-style de-duplication is sufficient here.
    """

    return (
        normalize_match_text(
            finding.get(
                "area",
                "",
            )
        ),

        normalize_match_text(
            finding.get(
                "evidence",
                "",
            )
        ),

        normalize_match_text(
            finding.get(
                "suggestion",
                "",
            )
        ),

        str(
            finding.get(
                "basis_type",
                "",
            )
        ).strip(),
    )


def merge_advisory_findings(
    first_findings: list[dict],
    second_findings: list[dict],
    first_judge: str,
    second_judge: str,
) -> list[dict]:
    """
    Merge optional advisory findings.

    Important:

    Advisory findings are NOT used to authorize mandatory actions.

    They remain optional Human-in-the-Loop guidance.
    """

    merged_map = {}

    for judge_key, findings in [
        (
            first_judge,
            first_findings,
        ),
        (
            second_judge,
            second_findings,
        ),
    ]:

        for finding in findings:

            identity = advisory_identity(
                finding
            )

            if identity not in merged_map:

                item = deepcopy(
                    finding
                )

                item[
                    "reported_by"
                ] = [
                    judge_key
                ]

                merged_map[
                    identity
                ] = item

            else:

                existing = (
                    merged_map[
                        identity
                    ]
                )

                reported_by = (
                    existing.get(
                        "reported_by",
                        [],
                    )
                    or []
                )

                if (
                    judge_key
                    not in reported_by
                ):

                    reported_by.append(
                        judge_key
                    )

                existing[
                    "reported_by"
                ] = (
                    reported_by
                )

    return list(
        merged_map.values()
    )


# =========================================================
# Review Notes Merge
# =========================================================

def merge_review_notes(
    first_notes: list,
    second_notes: list,
    first_judge: str,
    second_judge: str,
) -> list[str]:

    merged = []

    for judge_key, notes in [
        (
            first_judge,
            first_notes,
        ),
        (
            second_judge,
            second_notes,
        ),
    ]:

        for note in (
            notes or []
        ):

            note_text = str(
                note or ""
            ).strip()

            if not note_text:
                continue

            labeled_note = (
                f"[{judge_key}] "
                f"{note_text}"
            )

            if (
                labeled_note
                not in merged
            ):
                merged.append(
                    labeled_note
                )

    return merged


# =========================================================
# Cross-Judge Diagnostic Scores
# =========================================================

def average_diagnostic_scores(
    first_evaluation: dict,
    second_evaluation: dict,
) -> dict:
    """
    Average diagnostic scores across the two judges.

    These values remain diagnostic only.

    They must never be treated as calibrated pass/fail thresholds.
    """

    scores = {}

    for field in DIAGNOSTIC_FIELDS:

        first_value = (
            first_evaluation.get(
                field
            )
        )

        second_value = (
            second_evaluation.get(
                field
            )
        )

        numeric_values = []

        for value in [
            first_value,
            second_value,
        ]:

            if isinstance(
                value,
                (
                    int,
                    float,
                ),
            ):
                numeric_values.append(
                    float(
                        value
                    )
                )

        if numeric_values:

            scores[
                field
            ] = round(
                sum(
                    numeric_values
                )
                / len(
                    numeric_values
                ),
                1,
            )

    return scores


# =========================================================
# Single Judge Execution
# =========================================================

def run_single_judge(
    *,
    brand_info: str,
    campaign_brief: str,
    generated_content: str,
    policy_context: str,
    requirements,
    content_origin: str,
    judge_model_key: str,
) -> dict:
    """
    Run one evaluator instance explicitly using the supplied judge.
    """

    return evaluate_content(
        brand_info=
            brand_info,

        campaign_brief=
            campaign_brief,

        generated_content=
            generated_content,

        policy_context=
            policy_context,

        judge_model_key=
            judge_model_key,

        requirements=
            requirements,

        content_origin=
            content_origin,
    )


# =========================================================
# Fast Review
# =========================================================

def run_fast_review(
    *,
    brand_info: str,
    campaign_brief: str,
    generated_content: str,
    policy_context: str = "",
    requirements=None,
    content_origin: str = "generated",
    judge_model_key: str = DEFAULT_FAST_JUDGE,
) -> dict:
    """
    Interactive low-latency review using one Judge.
    """

    try:

        evaluation = (
            run_single_judge(
                brand_info=
                    brand_info,

                campaign_brief=
                    campaign_brief,

                generated_content=
                    generated_content,

                policy_context=
                    policy_context,

                requirements=
                    requirements,

                content_origin=
                    content_origin,

                judge_model_key=
                    judge_model_key,
            )
        )

    except Exception as error:

        return {
            "review_mode":
                FAST_REVIEW,

            "final_route":
                REVIEW_ERROR,

            "can_auto_fix":
                False,

            "requires_human_review":
                True,

            "evaluation":
                None,

            "judge_results":
                {},

            "judge_errors":
                {
                    judge_model_key:
                        str(
                            error
                        )
                },

            "cross_judge":
                None,
        }

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

        route = (
            COMPLIANCE_AND_REQUIREMENT_ACTION
        )

    elif compliance_findings:

        route = (
            COMPLIANCE_ACTION
        )

    elif requirement_findings:

        route = (
            REQUIREMENT_ACTION
        )

    else:

        route = (
            NO_MANDATORY_ACTION
        )

    return {
        "review_mode":
            FAST_REVIEW,

        "final_route":
            route,

        "can_auto_fix":
            route
            in {
                COMPLIANCE_ACTION,
                REQUIREMENT_ACTION,
                COMPLIANCE_AND_REQUIREMENT_ACTION,
            },

        "requires_human_review":
            False,

        "evaluation":
            evaluation,

        "judge_results":
            {
                judge_model_key:
                    evaluation
            },

        "judge_errors":
            {},

        "cross_judge":
            None,
    }


# =========================================================
# Cross-Judge Review
# =========================================================

def run_cross_judge_review(
    *,
    brand_info: str,
    campaign_brief: str,
    generated_content: str,
    policy_context: str = "",
    requirements=None,
    content_origin: str = "generated",
    judge_model_keys: tuple[str, str] = DEFAULT_CROSS_JUDGES,
) -> dict:
    """
    Run two independent judges and build a conservative consensus.

    Product rule:

    - Only independently confirmed mandatory findings are eligible
      for automatic correction.
    - Mandatory-layer disagreement escalates to Human Review.
    - Optional advisory disagreement does not trigger Human Review.
    - Diagnostic scores never control mandatory routing.
    """

    if len(
        judge_model_keys
    ) != 2:

        raise ValueError(
            "Cross-Judge Review currently requires exactly two judges."
        )

    first_judge = (
        judge_model_keys[0]
    )

    second_judge = (
        judge_model_keys[1]
    )

    judge_results = {}

    judge_errors = {}


    # -----------------------------------------------------
    # Judge 1
    # -----------------------------------------------------

    try:

        judge_results[
            first_judge
        ] = run_single_judge(
            brand_info=
                brand_info,

            campaign_brief=
                campaign_brief,

            generated_content=
                generated_content,

            policy_context=
                policy_context,

            requirements=
                requirements,

            content_origin=
                content_origin,

            judge_model_key=
                first_judge,
        )

    except Exception as error:

        judge_errors[
            first_judge
        ] = str(
            error
        )


    # -----------------------------------------------------
    # Judge 2
    # -----------------------------------------------------

    try:

        judge_results[
            second_judge
        ] = run_single_judge(
            brand_info=
                brand_info,

            campaign_brief=
                campaign_brief,

            generated_content=
                generated_content,

            policy_context=
                policy_context,

            requirements=
                requirements,

            content_origin=
                content_origin,

            judge_model_key=
                second_judge,
        )

    except Exception as error:

        judge_errors[
            second_judge
        ] = str(
            error
        )


    # -----------------------------------------------------
    # Judge Failure
    # -----------------------------------------------------

    if judge_errors:

        return {
            "review_mode":
                CROSS_JUDGE_REVIEW,

            "final_route":
                HUMAN_REVIEW_REQUIRED,

            "can_auto_fix":
                False,

            "requires_human_review":
                True,

            "evaluation":
                None,

            "judge_results":
                judge_results,

            "judge_errors":
                judge_errors,

            "cross_judge":
                {
                    "status":
                        "JUDGE_EXECUTION_INCOMPLETE",

                    "judges":
                        list(
                            judge_model_keys
                        ),

                    "message":
                        (
                            "Cross-Judge consensus could not be completed "
                            "because at least one Judge failed."
                        ),
                },
        }


    first_evaluation = (
        judge_results[
            first_judge
        ]
    )

    second_evaluation = (
        judge_results[
            second_judge
        ]
    )


    # -----------------------------------------------------
    # Compliance Consensus
    # -----------------------------------------------------

    (
        consensus_compliance,
        first_only_compliance,
        second_only_compliance,
    ) = match_compliance_findings(
        first_findings=(
            first_evaluation.get(
                "compliance_findings",
                [],
            )
            or []
        ),

        second_findings=(
            second_evaluation.get(
                "compliance_findings",
                [],
            )
            or []
        ),

        first_judge=
            first_judge,

        second_judge=
            second_judge,
    )


    # -----------------------------------------------------
    # Requirement Consensus
    # -----------------------------------------------------

    (
        consensus_requirements,
        first_only_requirements,
        second_only_requirements,
    ) = match_requirement_findings(
        first_findings=(
            first_evaluation.get(
                "requirement_findings",
                [],
            )
            or []
        ),

        second_findings=(
            second_evaluation.get(
                "requirement_findings",
                [],
            )
            or []
        ),

        first_judge=
            first_judge,

        second_judge=
            second_judge,
    )


    # -----------------------------------------------------
    # Advisory Merge
    # -----------------------------------------------------

    merged_advisories = (
        merge_advisory_findings(
            first_findings=(
                first_evaluation.get(
                    "advisory_findings",
                    [],
                )
                or []
            ),

            second_findings=(
                second_evaluation.get(
                    "advisory_findings",
                    [],
                )
                or []
            ),

            first_judge=
                first_judge,

            second_judge=
                second_judge,
        )
    )


    # -----------------------------------------------------
    # Mandatory-Layer Disagreement
    # -----------------------------------------------------

    compliance_disagreement = bool(
        first_only_compliance
        or second_only_compliance
    )

    requirement_disagreement = bool(
        first_only_requirements
        or second_only_requirements
    )

    mandatory_disagreement = bool(
        compliance_disagreement
        or requirement_disagreement
    )


    # -----------------------------------------------------
    # Final Routing
    # -----------------------------------------------------

    if mandatory_disagreement:

        final_route = (
            HUMAN_REVIEW_REQUIRED
        )

        can_auto_fix = False

        requires_human_review = (
            True
        )

    elif (
        consensus_compliance
        and consensus_requirements
    ):

        final_route = (
            COMPLIANCE_AND_REQUIREMENT_ACTION
        )

        can_auto_fix = True

        requires_human_review = (
            False
        )

    elif consensus_compliance:

        final_route = (
            COMPLIANCE_ACTION
        )

        can_auto_fix = True

        requires_human_review = (
            False
        )

    elif consensus_requirements:

        final_route = (
            REQUIREMENT_ACTION
        )

        can_auto_fix = True

        requires_human_review = (
            False
        )

    else:

        final_route = (
            NO_MANDATORY_ACTION
        )

        can_auto_fix = False

        requires_human_review = (
            False
        )


    # -----------------------------------------------------
    # Consensus Evaluation
    # -----------------------------------------------------

    consensus_evaluation = deepcopy(
        first_evaluation
    )

    consensus_evaluation[
        "compliance_findings"
    ] = consensus_compliance

    consensus_evaluation[
        "requirement_findings"
    ] = consensus_requirements

    consensus_evaluation[
        "advisory_findings"
    ] = merged_advisories

    consensus_evaluation[
        "review_notes"
    ] = merge_review_notes(
        first_notes=(
            first_evaluation.get(
                "review_notes",
                [],
            )
            or []
        ),

        second_notes=(
            second_evaluation.get(
                "review_notes",
                [],
            )
            or []
        ),

        first_judge=
            first_judge,

        second_judge=
            second_judge,
    )


    diagnostic_scores = (
        average_diagnostic_scores(
            first_evaluation,
            second_evaluation,
        )
    )

    consensus_evaluation.update(
        diagnostic_scores
    )

    consensus_evaluation[
        "blocking_count"
    ] = len(
        consensus_compliance
    )

    consensus_evaluation[
        "requirement_count"
    ] = len(
        consensus_requirements
    )

    consensus_evaluation[
        "cross_judge_consensus"
    ] = True

    consensus_evaluation[
        "cross_judge_final_route"
    ] = final_route

    consensus_evaluation[
        "requires_human_review"
    ] = requires_human_review


    # -----------------------------------------------------
    # Cross-Judge Diagnostics
    # -----------------------------------------------------

    cross_judge_details = {
        "status":
            (
                "MANDATORY_DISAGREEMENT"
                if mandatory_disagreement
                else "CONSENSUS_COMPLETE"
            ),

        "judges":
            [
                first_judge,
                second_judge,
            ],

        "consensus_compliance_count":
            len(
                consensus_compliance
            ),

        "consensus_requirement_count":
            len(
                consensus_requirements
            ),

        "compliance_disagreement":
            compliance_disagreement,

        "requirement_disagreement":
            requirement_disagreement,

        "mandatory_disagreement":
            mandatory_disagreement,

        "first_only_compliance":
            first_only_compliance,

        "second_only_compliance":
            second_only_compliance,

        "first_only_requirements":
            first_only_requirements,

        "second_only_requirements":
            second_only_requirements,
    }


    return {
        "review_mode":
            CROSS_JUDGE_REVIEW,

        "final_route":
            final_route,

        "can_auto_fix":
            can_auto_fix,

        "requires_human_review":
            requires_human_review,

        "evaluation":
            consensus_evaluation,

        "judge_results":
            judge_results,

        "judge_errors":
            {},

        "cross_judge":
            cross_judge_details,
    }


# =========================================================
# Public Product Review API
# =========================================================

def review_content(
    *,
    brand_info: str,
    campaign_brief: str,
    generated_content: str,
    policy_context: str = "",
    requirements=None,
    content_origin: str = "generated",
    review_mode: str = FAST_REVIEW,
    fast_judge_model_key: str = DEFAULT_FAST_JUDGE,
    cross_judge_model_keys: tuple[str, str] = DEFAULT_CROSS_JUDGES,
) -> dict:
    """
    Main product-facing review interface.

    Parameters
    ----------
    review_mode:
        "fast"
        or
        "cross_judge"

    Fast Review
    -----------
    Uses one Judge for responsiveness.

    Cross-Judge Review
    ------------------
    Uses two independent Judges and deterministic consensus routing.

    Important
    ---------
    This orchestration layer does not modify evaluator.py.

    evaluator.py remains the single-judge evaluation engine.
    """

    normalized_mode = str(
        review_mode
        or FAST_REVIEW
    ).strip().lower()

    if normalized_mode == FAST_REVIEW:

        return run_fast_review(
            brand_info=
                brand_info,

            campaign_brief=
                campaign_brief,

            generated_content=
                generated_content,

            policy_context=
                policy_context,

            requirements=
                requirements,

            content_origin=
                content_origin,

            judge_model_key=
                fast_judge_model_key,
        )

    if normalized_mode in {
        CROSS_JUDGE_REVIEW,
        "cross",
        "cross-judge",
        "cross_judge_review",
    }:

        return run_cross_judge_review(
            brand_info=
                brand_info,

            campaign_brief=
                campaign_brief,

            generated_content=
                generated_content,

            policy_context=
                policy_context,

            requirements=
                requirements,

            content_origin=
                content_origin,

            judge_model_keys=
                cross_judge_model_keys,
        )

    raise ValueError(
        (
            "Unsupported review_mode: "
            f"{review_mode}. "
            "Use 'fast' or 'cross_judge'."
        )
    )