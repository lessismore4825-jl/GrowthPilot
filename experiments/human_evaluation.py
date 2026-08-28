from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from src.reviser import fix_mandatory_issues, optimize_quality
from src.review_orchestrator import (
    CROSS_JUDGE_REVIEW,
    HUMAN_REVIEW_REQUIRED,
    NO_MANDATORY_ACTION,
    REVIEW_ERROR,
    review_content,
)


# =========================================================
# Configuration
# =========================================================

RUNNER_VERSION = "2.0"

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "human_eval_cases.json"

# IMPORTANT:
# Use a brand-new result directory so stale v1 checkpoints can never
# contaminate this rerun.
RESULT_DIR = ROOT / "experiments" / "results" / "human_eval_v2"

CHECKPOINT_FILE = RESULT_DIR / "checkpoint.json"
FINAL_FILE = RESULT_DIR / "human_eval_results.json"
BLIND_PACKET_FILE = RESULT_DIR / "human_eval_blind_packet.json"
RATINGS_TEMPLATE_FILE = RESULT_DIR / "human_eval_ratings_template.csv"


class ReviewInfrastructureError(RuntimeError):
    """Raised when a judge/provider execution failure invalidates a review run."""


# =========================================================
# Basic IO
# =========================================================

def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


# =========================================================
# Context Construction
# Mirrors the current Streamlit app as closely as possible.
# =========================================================

def build_brand_context(
    case: dict,
) -> str:

    brand = str(
        case.get("brand") or ""
    ).strip()

    context = (
        case.get("brand_context")
        or {}
    )

    facts = (
        case.get(
            "verified_product_information"
        )
        or []
    )

    restrictions = (
        context.get("restrictions")
        or []
    )

    restriction_text = "\n".join(
        f"- {item}"
        for item in restrictions
    ) or "- None specified"

    facts_text = "\n".join(
        f"- {item}"
        for item in facts
    ) or "- None specified"

    return (
        f"Brand: {brand}\n\n"
        f"Brand positioning:\n"
        f"{context.get('positioning', '')}\n\n"
        f"Brand tone:\n"
        f"{context.get('tone', '')}\n\n"
        f"Restrictions:\n"
        f"{restriction_text}\n\n"
        "VERIFIED PRODUCT INFORMATION:\n"
        f"{facts_text}"
    ).strip()


def normalized_requirements(
    case: dict,
) -> list[dict]:

    result = []

    for item in (
        case.get("must_mention")
        or []
    ):

        content = str(
            item.get("content") or ""
        ).strip()

        if not content:
            continue

        result.append(
            {
                "requirement_id":
                    str(
                        item.get(
                            "requirement_id"
                        )
                        or ""
                    ).strip(),

                "content":
                    content,

                "match_mode":
                    str(
                        item.get(
                            "match_mode"
                        )
                        or "SEMANTIC"
                    )
                    .strip()
                    .upper(),
            }
        )

    return result


def build_campaign_context(
    case: dict,
) -> str:

    reqs = normalized_requirements(
        case
    )

    must_avoid = (
        case.get("must_avoid")
        or []
    )

    # Mirror app.py:
    # the Campaign Context shows the requirement CONTENT only.
    # IDs / EXACT / SEMANTIC stay in the separate structured
    # `requirements=` argument passed to the evaluator.
    requirement_text = "\n".join(
        f"- {item['content']}"
        for item in reqs
    ) or "- None specified"

    avoid_text = "\n".join(
        f"- {item}"
        for item in must_avoid
    ) or "- None specified"

    custom_brief = str(
        case.get("campaign_brief")
        or ""
    ).strip()

    if not custom_brief:
        custom_brief = (
            f"Objective:\n"
            f"Introduce {case.get('brand', '')} "
            "using only verified product facts.\n\n"
            "Campaign approach:\n"
            "Follow the supplied Must Mention and Must Avoid "
            "requirements. Keep the content appropriate for the "
            "supplied platform, content type, brand tone, and "
            "creator context."
        )

    creator_context = str(
        case.get("creator_context")
        or "No creator profile supplied"
    ).strip()

    return (
        "ORIGINAL CAMPAIGN BRIEF:\n"
        f"{custom_brief}\n\n"
        "PLATFORM:\n"
        f"{case.get('platform', 'Not specified')}\n\n"
        "CONTENT TYPE:\n"
        f"{case.get('content_type', 'Not specified')}\n\n"
        "CREATOR PROFILE:\n"
        f"- {creator_context}\n\n"
        "CAMPAIGN MUST MENTION:\n"
        f"{requirement_text}\n\n"
        "CAMPAIGN MUST AVOID:\n"
        f"{avoid_text}"
    ).strip()


def build_policy_context(
    case: dict,
) -> str:

    return str(
        case.get("policy_context")
        or ""
    ).strip()


# =========================================================
# Review Validation
# =========================================================

def validate_cross_judge_result(
    result: dict,
    *,
    stage: str,
) -> None:
    """
    Critical benchmark guardrail.

    Production behavior is allowed to route a judge execution failure
    to HUMAN_REVIEW_REQUIRED for safety.

    Evaluation behavior is different:
    a provider / judge failure is NOT a valid evaluated case.
    It must trigger retry instead of being checkpointed as success.
    """

    if not isinstance(
        result,
        dict,
    ):
        raise ReviewInfrastructureError(
            f"{stage}: review_content returned "
            f"{type(result).__name__}, expected dict."
        )

    judge_errors = (
        result.get("judge_errors")
        or {}
    )

    if judge_errors:

        readable_errors = "; ".join(
            f"{judge}: {message}"
            for judge, message
            in judge_errors.items()
        )

        raise ReviewInfrastructureError(
            f"{stage}: judge execution error(s): "
            f"{readable_errors}"
        )

    judge_results = (
        result.get("judge_results")
        or {}
    )

    missing_judges = [
        judge
        for judge in (
            "step",
            "qwen",
        )
        if not judge_results.get(
            judge
        )
    ]

    if missing_judges:
        raise ReviewInfrastructureError(
            f"{stage}: missing judge result(s): "
            + ", ".join(
                missing_judges
            )
        )

    if (
        result.get("final_route")
        == REVIEW_ERROR
    ):
        raise ReviewInfrastructureError(
            f"{stage}: final route is REVIEW_ERROR."
        )


def compact_evaluation(
    evaluation: dict | None,
) -> dict:

    evaluation = (
        evaluation
        or {}
    )

    return {
        "compliance_findings":
            evaluation.get(
                "compliance_findings",
                [],
            )
            or [],

        "requirement_findings":
            evaluation.get(
                "requirement_findings",
                [],
            )
            or [],

        "advisory_findings":
            evaluation.get(
                "advisory_findings",
                [],
            )
            or [],

        "review_notes":
            evaluation.get(
                "review_notes",
                [],
            )
            or [],

        "brand_alignment":
            evaluation.get(
                "brand_alignment"
            ),

        "tone_match":
            evaluation.get(
                "tone_match"
            ),

        "selling_point_coverage":
            evaluation.get(
                "selling_point_coverage"
            ),

        "factual_consistency":
            evaluation.get(
                "factual_consistency"
            ),

        "unsupported_claim_risk":
            evaluation.get(
                "unsupported_claim_risk"
            ),

        "heuristic_composite_score":
            evaluation.get(
                "heuristic_composite_score"
            ),
    }


def perform_review(
    case: dict,
    content: str,
    *,
    stage: str,
) -> tuple[dict, float]:

    start = time.perf_counter()

    result = review_content(
        brand_info=
            build_brand_context(
                case
            ),

        campaign_brief=
            build_campaign_context(
                case
            ),

        generated_content=
            content,

        policy_context=
            build_policy_context(
                case
            ),

        requirements=
            normalized_requirements(
                case
            ),

        content_origin=
            "creator_draft",

        review_mode=
            CROSS_JUDGE_REVIEW,
    )

    latency = round(
        time.perf_counter()
        - start,
        2,
    )

    validate_cross_judge_result(
        result,
        stage=stage,
    )

    return (
        result,
        latency,
    )


# =========================================================
# Retry
# =========================================================

def call_with_retry(
    fn: Callable,
    *,
    retries: int,
    wait_seconds: int,
    label: str,
):

    last_error = None

    for attempt in range(
        1,
        retries + 2,
    ):

        try:
            return fn()

        except Exception as error:

            last_error = error

            print(
                f"  {label} attempt "
                f"{attempt}/{retries + 1} failed:"
            )
            print(
                f"    {type(error).__name__}: "
                f"{error}"
            )

            if attempt > retries:
                raise

            sleep_seconds = (
                wait_seconds
                * attempt
            )

            print(
                f"  retrying {label} "
                f"in {sleep_seconds}s..."
            )

            time.sleep(
                sleep_seconds
            )

    raise last_error


# =========================================================
# Revision
# =========================================================

def choose_revision_action(
    case: dict,
    review_result: dict,
) -> str | None:

    if review_result.get(
        "requires_human_review",
        False,
    ):
        return None

    if review_result.get(
        "can_auto_fix",
        False,
    ):
        return "mandatory"

    evaluation = (
        review_result.get(
            "evaluation"
        )
        or {}
    )

    advisories = (
        evaluation.get(
            "advisory_findings"
        )
        or []
    )

    # Evaluation protocol:
    # Only cases pre-designated as advisory-only simulate
    # a user voluntarily clicking Optional Optimization.
    # This does NOT affect original routing evaluation.
    if (
        case.get("category")
        == "advisory_only"
        and advisories
    ):
        return "optional"

    return None


def revise_content(
    *,
    case: dict,
    original_content: str,
    review_result: dict,
    action: str,
    model_key: str,
) -> tuple[str, float]:

    evaluation = (
        review_result.get(
            "evaluation"
        )
        or {}
    )

    start = time.perf_counter()

    if action == "mandatory":

        revised = (
            fix_mandatory_issues(
                brand_info=
                    build_brand_context(
                        case
                    ),

                campaign_brief=
                    build_campaign_context(
                        case
                    ),

                original_content=
                    original_content,

                evaluation=
                    evaluation,

                policy_context=
                    build_policy_context(
                        case
                    ),

                model_key=
                    model_key,
            )
        )

    elif action == "optional":

        revised = (
            optimize_quality(
                brand_info=
                    build_brand_context(
                        case
                    ),

                campaign_brief=
                    build_campaign_context(
                        case
                    ),

                original_content=
                    original_content,

                evaluation=
                    evaluation,

                policy_context=
                    build_policy_context(
                        case
                    ),

                model_key=
                    model_key,
            )
        )

    else:

        raise ValueError(
            f"Unsupported revision action: "
            f"{action}"
        )

    latency = round(
        time.perf_counter()
        - start,
        2,
    )

    return (
        revised,
        latency,
    )


# =========================================================
# Diagnostics
# =========================================================

def finding_counts(
    evaluation: dict,
) -> tuple[int, int, int]:

    return (
        len(
            evaluation.get(
                "compliance_findings",
                [],
            )
            or []
        ),

        len(
            evaluation.get(
                "requirement_findings",
                [],
            )
            or []
        ),

        len(
            evaluation.get(
                "advisory_findings",
                [],
            )
            or []
        ),
    )


def print_judge_diagnostics(
    review_result: dict,
) -> None:

    judge_results = (
        review_result.get(
            "judge_results"
        )
        or {}
    )

    for judge_key in (
        "step",
        "qwen",
    ):

        judge_result = (
            judge_results.get(
                judge_key
            )
            or {}
        )

        compliance, requirement, advisory = (
            finding_counts(
                judge_result
            )
        )

        print(
            f"  {judge_key}: "
            f"{compliance} compliance / "
            f"{requirement} requirement / "
            f"{advisory} advisory"
        )

    cross = (
        review_result.get(
            "cross_judge"
        )
        or {}
    )

    if review_result.get(
        "final_route"
    ) == HUMAN_REVIEW_REQUIRED:

        first_c = (
            cross.get(
                "first_only_compliance"
            )
            or []
        )

        second_c = (
            cross.get(
                "second_only_compliance"
            )
            or []
        )

        first_r = (
            cross.get(
                "first_only_requirements"
            )
            or []
        )

        second_r = (
            cross.get(
                "second_only_requirements"
            )
            or []
        )

        print(
            "  disagreement details: "
            f"{len(first_c)} first-only compliance / "
            f"{len(second_c)} second-only compliance / "
            f"{len(first_r)} first-only requirement / "
            f"{len(second_r)} second-only requirement"
        )


# =========================================================
# Run One Case
# =========================================================

def run_case(
    case: dict,
    *,
    revision_model: str,
    retries: int,
    retry_wait: int,
) -> dict:

    case_id = str(
        case["case_id"]
    )

    draft = str(
        case["creator_draft"]
    )

    print(
        f"\n{'=' * 78}\n"
        f"{case_id} | "
        f"{case.get('category')} | "
        f"{case.get('brand')}\n"
        f"{'=' * 78}"
    )

    review_result, review_latency = (
        call_with_retry(
            lambda: perform_review(
                case,
                draft,
                stage=(
                    f"{case_id} original review"
                ),
            ),
            retries=retries,
            wait_seconds=retry_wait,
            label=(
                f"{case_id} original review"
            ),
        )
    )

    evaluation = (
        review_result.get(
            "evaluation"
        )
        or {}
    )

    compliance, requirement, advisory = (
        finding_counts(
            evaluation
        )
    )

    print(
        "  original route:",
        review_result.get(
            "final_route"
        ),
    )

    print(
        "  consensus findings:",
        f"{compliance} compliance / "
        f"{requirement} requirement / "
        f"{advisory} advisory",
    )

    print_judge_diagnostics(
        review_result
    )

    print(
        f"  review latency: "
        f"{review_latency:.2f}s"
    )

    action = (
        choose_revision_action(
            case,
            review_result,
        )
    )

    revised_content = None
    revision_latency = None

    revised_review_result = None
    revised_review_latency = None

    if action:

        print(
            f"  simulated user action: "
            f"{action}"
        )

        revised_content, revision_latency = (
            call_with_retry(
                lambda: revise_content(
                    case=case,
                    original_content=draft,
                    review_result=
                        review_result,
                    action=action,
                    model_key=
                        revision_model,
                ),
                retries=retries,
                wait_seconds=retry_wait,
                label=(
                    f"{case_id} revision"
                ),
            )
        )

        revised_review_result, revised_review_latency = (
            call_with_retry(
                lambda: perform_review(
                    case,
                    revised_content,
                    stage=(
                        f"{case_id} recheck"
                    ),
                ),
                retries=retries,
                wait_seconds=retry_wait,
                label=(
                    f"{case_id} recheck"
                ),
            )
        )

        print(
            "  revised route:",
            revised_review_result.get(
                "final_route"
            ),
        )

        print_judge_diagnostics(
            revised_review_result
        )

        print(
            f"  revision latency: "
            f"{revision_latency:.2f}s"
        )

        print(
            f"  revised review latency: "
            f"{revised_review_latency:.2f}s"
        )

    else:

        if review_result.get(
            "requires_human_review",
            False,
        ):

            print(
                "  no revision: "
                "Human Review gate is active"
            )

        else:

            print(
                "  no revision: "
                "no simulated action required"
            )

    return {
        "case_id":
            case_id,

        "category":
            case.get("category"),

        "brand":
            case.get("brand"),

        "designed_gold":
            case.get(
                "designed_gold"
            ),

        "resolved_context": {
            "brand_context":
                build_brand_context(
                    case
                ),

            "campaign_context":
                build_campaign_context(
                    case
                ),

            "policy_context":
                build_policy_context(
                    case
                ),

            "requirements":
                normalized_requirements(
                    case
                ),
        },

        "input": {
            "platform":
                case.get("platform"),

            "content_type":
                case.get(
                    "content_type"
                ),

            "brand_context":
                case.get(
                    "brand_context"
                ),

            "verified_product_information":
                case.get(
                    "verified_product_information"
                ),

            "must_mention":
                case.get(
                    "must_mention"
                ),

            "must_avoid":
                case.get(
                    "must_avoid"
                ),

            "creator_context":
                case.get(
                    "creator_context"
                ),

            "creator_draft":
                draft,
        },

        "original": {
            "review_latency_seconds":
                review_latency,

            "final_route":
                review_result.get(
                    "final_route"
                ),

            "requires_human_review":
                review_result.get(
                    "requires_human_review",
                    False,
                ),

            "can_auto_fix":
                review_result.get(
                    "can_auto_fix",
                    False,
                ),

            "evaluation":
                compact_evaluation(
                    evaluation
                ),

            "judge_results":
                review_result.get(
                    "judge_results",
                    {},
                ),

            "cross_judge":
                review_result.get(
                    "cross_judge",
                    {},
                ),

            "judge_errors":
                review_result.get(
                    "judge_errors",
                    {},
                ),
        },

        "revision": {
            "action":
                action,

            "content":
                revised_content,

            "revision_latency_seconds":
                revision_latency,

            "recheck_latency_seconds":
                revised_review_latency,

            "recheck_final_route":
                (
                    revised_review_result
                    .get(
                        "final_route"
                    )
                    if revised_review_result
                    else None
                ),

            "recheck_requires_human_review":
                (
                    revised_review_result
                    .get(
                        "requires_human_review",
                        False,
                    )
                    if revised_review_result
                    else None
                ),

            "recheck_evaluation":
                (
                    compact_evaluation(
                        revised_review_result
                        .get(
                            "evaluation"
                        )
                    )
                    if revised_review_result
                    else None
                ),

            "recheck_judge_results":
                (
                    revised_review_result
                    .get(
                        "judge_results",
                        {},
                    )
                    if revised_review_result
                    else None
                ),

            "recheck_cross_judge":
                (
                    revised_review_result
                    .get(
                        "cross_judge",
                        {},
                    )
                    if revised_review_result
                    else None
                ),

            "recheck_judge_errors":
                (
                    revised_review_result
                    .get(
                        "judge_errors",
                        {},
                    )
                    if revised_review_result
                    else {}
                ),
        },
    }


# =========================================================
# Blind Packet
# =========================================================

def anonymize_finding(
    finding: dict,
) -> dict:

    if not isinstance(
        finding,
        dict,
    ):
        return {
            "value":
                str(finding)
        }

    blocked_keys = {
        "reported_by",
        "judge_evidence",
        "judge_reasons",
    }

    return {
        key: value
        for key, value
        in finding.items()
        if key not in blocked_keys
    }


def anonymized_disagreement_details(
    cross_judge: dict,
) -> dict:

    cross_judge = (
        cross_judge
        or {}
    )

    compliance = []

    for key in (
        "first_only_compliance",
        "second_only_compliance",
    ):
        for finding in (
            cross_judge.get(
                key
            )
            or []
        ):
            compliance.append(
                anonymize_finding(
                    finding
                )
            )

    requirements = []

    for key in (
        "first_only_requirements",
        "second_only_requirements",
    ):
        for finding in (
            cross_judge.get(
                key
            )
            or []
        ):
            requirements.append(
                anonymize_finding(
                    finding
                )
            )

    return {
        "disputed_compliance_findings":
            compliance,

        "disputed_requirement_findings":
            requirements,
    }


def create_blind_packet(
    dataset: dict,
    result_records: list[dict],
) -> dict:

    by_id = {
        item["case_id"]: item
        for item in result_records
    }

    blind_cases = []

    for case in dataset["cases"]:

        case_id = case["case_id"]

        if case_id not in by_id:
            continue

        record = by_id[
            case_id
        ]

        original = record[
            "original"
        ]

        revision = record[
            "revision"
        ]

        disagreement = (
            anonymized_disagreement_details(
                original.get(
                    "cross_judge"
                )
                or {}
            )
        )

        blind_cases.append(
            {
                "case_id":
                    case_id,

                "campaign_context": {
                    "brand":
                        case["brand"],

                    "platform":
                        case.get(
                            "platform"
                        ),

                    "content_type":
                        case.get(
                            "content_type"
                        ),

                    "brand_context":
                        case.get(
                            "brand_context"
                        ),

                    "verified_product_information":
                        case.get(
                            "verified_product_information"
                        ),

                    "must_mention":
                        case.get(
                            "must_mention"
                        ),

                    "must_avoid":
                        case.get(
                            "must_avoid"
                        ),

                    "creator_context":
                        case.get(
                            "creator_context"
                        ),
                },

                "creator_draft":
                    case[
                        "creator_draft"
                    ],

                "system_output": {
                    # Designed labels, judge identities, and
                    # diagnostic scores are intentionally omitted.
                    "final_route":
                        original.get(
                            "final_route"
                        ),

                    "requires_human_review":
                        original.get(
                            "requires_human_review"
                        ),

                    "compliance_findings":
                        original[
                            "evaluation"
                        ].get(
                            "compliance_findings",
                            [],
                        ),

                    "requirement_findings":
                        original[
                            "evaluation"
                        ].get(
                            "requirement_findings",
                            [],
                        ),

                    "advisory_findings":
                        original[
                            "evaluation"
                        ].get(
                            "advisory_findings",
                            [],
                        ),

                    # If Human Review is triggered, preserve the
                    # disputed mandatory content but remove judge
                    # identity. This makes blind evaluation useful
                    # without revealing Step vs Qwen.
                    "human_review_disagreement":
                        disagreement,

                    "revision_action":
                        revision.get(
                            "action"
                        ),

                    "revised_content":
                        revision.get(
                            "content"
                        ),

                    "recheck_final_route":
                        revision.get(
                            "recheck_final_route"
                        ),

                    "recheck_compliance_findings":
                        (
                            (
                                revision.get(
                                    "recheck_evaluation"
                                )
                                or {}
                            ).get(
                                "compliance_findings",
                                [],
                            )
                        ),

                    "recheck_requirement_findings":
                        (
                            (
                                revision.get(
                                    "recheck_evaluation"
                                )
                                or {}
                            ).get(
                                "requirement_findings",
                                [],
                            )
                        ),

                    "recheck_advisory_findings":
                        (
                            (
                                revision.get(
                                    "recheck_evaluation"
                                )
                                or {}
                            ).get(
                                "advisory_findings",
                                [],
                            )
                        ),
                },
            }
        )

    return {
        "evaluation_name":
            "GrowthPilot Blind AI-assisted Evaluation Packet",

        "runner_version":
            RUNNER_VERSION,

        "important_note":
            (
                "Designed labels, judge identities, and diagnostic "
                "scores are intentionally omitted to reduce reviewer "
                "bias. Human-Review disagreement evidence is retained "
                "without judge identity."
            ),

        "rubric": {
            "finding_correctness_1_5":
                (
                    "Are the identified issues / non-issues "
                    "reasonable?"
                ),

            "finding_usefulness_1_5":
                (
                    "Would the findings help a campaign reviewer "
                    "make a decision?"
                ),

            "revision_correctness_1_5_or_na":
                (
                    "If a revision exists, did it correctly address "
                    "the intended issue without introducing a new one?"
                ),

            "voice_preservation_1_5_or_na":
                (
                    "If a revision exists, did it avoid unnecessary "
                    "rewriting of creator expression?"
                ),

            "advisory_usefulness_1_5_or_na":
                (
                    "If advisories exist, are they worth considering?"
                ),

            "mandatory_action_appropriate_yes_no":
                (
                    "Was the system right to require, not require, "
                    "or escalate mandatory action?"
                ),

            "over_editing_yes_no_or_na":
                (
                    "If a revision exists, did it change more "
                    "than necessary?"
                ),

            "accept_output_yes_no":
                (
                    "Would you accept this system output as a useful "
                    "pre-review result?"
                ),
        },

        "cases":
            blind_cases,
    }


# =========================================================
# Ratings Template
# =========================================================

def write_ratings_template(
    path: Path,
    result_records: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rater_id",
        "case_id",
        "finding_correctness_1_5",
        "finding_usefulness_1_5",
        "revision_correctness_1_5_or_na",
        "voice_preservation_1_5_or_na",
        "advisory_usefulness_1_5_or_na",
        "mandatory_action_appropriate_yes_no",
        "over_editing_yes_no_or_na",
        "accept_output_yes_no",
        "notes",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in result_records:

            writer.writerow(
                {
                    "rater_id":
                        "",

                    "case_id":
                        record[
                            "case_id"
                        ],
                }
            )


# =========================================================
# Automated Summary
# =========================================================

def automated_summary(
    records: list[dict],
) -> dict:

    determinate = []
    boundary = []
    revised = []

    for record in records:

        expected = (
            record.get(
                "designed_gold"
            )
            or {}
        ).get(
            "expected_route"
        )

        actual = (
            record.get(
                "original"
            )
            or {}
        ).get(
            "final_route"
        )

        if expected == "BOUNDARY_CASE":

            boundary.append(
                {
                    "case_id":
                        record[
                            "case_id"
                        ],

                    "actual_route":
                        actual,
                }
            )

        else:

            determinate.append(
                {
                    "case_id":
                        record[
                            "case_id"
                        ],

                    "expected_route":
                        expected,

                    "actual_route":
                        actual,

                    "route_match":
                        expected == actual,
                }
            )

        revision = (
            record.get(
                "revision"
            )
            or {}
        )

        if revision.get(
            "action"
        ):

            revised.append(
                {
                    "case_id":
                        record[
                            "case_id"
                        ],

                    "action":
                        revision.get(
                            "action"
                        ),

                    "recheck_route":
                        revision.get(
                            "recheck_final_route"
                        ),

                    "cleared_to_no_mandatory":
                        (
                            revision.get(
                                "recheck_final_route"
                            )
                            == NO_MANDATORY_ACTION
                        ),
                }
            )

    route_matches = sum(
        1
        for item in determinate
        if item[
            "route_match"
        ]
    )

    return {
        "completed_valid_cases":
            len(records),

        "determinate_route_match": {
            "matched":
                route_matches,

            "total":
                len(
                    determinate
                ),

            "rate":
                (
                    route_matches
                    / len(
                        determinate
                    )
                    if determinate
                    else None
                ),

            "cases":
                determinate,
        },

        "boundary_case_routes":
            boundary,

        "revision_recheck":
            revised,

        "important_note":
            (
                "Provider / judge execution failures are excluded "
                "from valid cases and must be rerun."
            ),
    }


# =========================================================
# CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the 20-case held-out GrowthPilot "
            "Cross-Judge evaluation suite."
        )
    )

    parser.add_argument(
        "--revision-model",
        default="step",
        choices=[
            "step",
            "qwen",
        ],
        help=(
            "Model used for simulated mandatory fix "
            "or advisory-only optional optimization."
        ),
    )

    parser.add_argument(
        "--cases",
        default="",
        help=(
            "Optional comma-separated case IDs, "
            "e.g. H01,H07,H19. "
            "Empty means all cases."
        ),
    )

    parser.add_argument(
        "--case-retries",
        type=int,
        default=3,
        help=(
            "Retries after judge/provider execution failures. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--retry-wait",
        type=int,
        default=15,
        help=(
            "Base seconds before retry. Backoff increases "
            "with each failed attempt. Default: 15."
        ),
    )

    parser.add_argument(
        "--inter-case-wait",
        type=int,
        default=5,
        help=(
            "Seconds to wait between completed cases. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore v2 checkpoint and rerun selected cases."
        ),
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    dataset = load_json(
        DATA_FILE
    )

    selected_ids = {
        item.strip()
        for item
        in args.cases.split(",")
        if item.strip()
    }

    selected_cases = [
        case
        for case
        in dataset["cases"]
        if (
            not selected_ids
            or case[
                "case_id"
            ]
            in selected_ids
        )
    ]

    if not selected_cases:

        raise SystemExit(
            "No matching cases selected."
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "evaluation_name":
            dataset[
                "benchmark_name"
            ],

        "dataset_version":
            dataset[
                "version"
            ],

        "runner_version":
            RUNNER_VERSION,

        "records":
            {},
    }

    if (
        CHECKPOINT_FILE.exists()
        and not args.fresh
    ):

        loaded_checkpoint = (
            load_json(
                CHECKPOINT_FILE
            )
        )

        if (
            loaded_checkpoint.get(
                "runner_version"
            )
            == RUNNER_VERSION
            and loaded_checkpoint.get(
                "dataset_version"
            )
            == dataset[
                "version"
            ]
        ):

            checkpoint = (
                loaded_checkpoint
            )

        else:

            print(
                "Existing checkpoint has a different "
                "runner/dataset version; starting a clean v2 run."
            )

    print(
        f"Runner version: "
        f"{RUNNER_VERSION}"
    )

    print(
        f"Selected "
        f"{len(selected_cases)} "
        f"case(s)."
    )

    print(
        "Cross-Judge mode: "
        "Step + Qwen"
    )

    print(
        f"Revision model: "
        f"{args.revision_model}"
    )

    print(
        f"Result directory: "
        f"{RESULT_DIR}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_FILE}"
    )

    failed_case_ids = []

    for index, case in enumerate(
        selected_cases,
        start=1,
    ):

        case_id = case[
            "case_id"
        ]

        previous = (
            checkpoint.get(
                "records",
                {},
            ).get(
                case_id
            )
        )

        # Only a genuinely valid previous record is skippable.
        # Infrastructure-failed records are retried automatically.
        if (
            previous
            and "error" not in previous
            and not args.fresh
        ):

            print(
                f"\n"
                f"[{index}/"
                f"{len(selected_cases)}] "
                f"{case_id} "
                "already completed — skip"
            )

            continue

        print(
            f"\n"
            f"[{index}/"
            f"{len(selected_cases)}] "
            f"running {case_id}"
        )

        try:

            record = run_case(
                case,
                revision_model=
                    args.revision_model,
                retries=
                    args.case_retries,
                retry_wait=
                    args.retry_wait,
            )

        except Exception as error:

            failed_case_ids.append(
                case_id
            )

            print(
                f"  CASE INVALID / FAILED: "
                f"{error}"
            )

            traceback.print_exc()

            record = {
                "case_id":
                    case_id,

                "category":
                    case.get(
                        "category"
                    ),

                "brand":
                    case.get(
                        "brand"
                    ),

                "designed_gold":
                    case.get(
                        "designed_gold"
                    ),

                "error":
                    str(
                        error
                    ),

                "error_type":
                    type(
                        error
                    ).__name__,

                "traceback":
                    traceback.format_exc(),
            }

        checkpoint.setdefault(
            "records",
            {},
        )[case_id] = record

        save_json(
            CHECKPOINT_FILE,
            checkpoint,
        )

        save_json(
            RESULT_DIR
            / f"{case_id}_result.json",
            record,
        )

        if (
            index
            < len(
                selected_cases
            )
            and args.inter_case_wait > 0
        ):

            print(
                f"  waiting "
                f"{args.inter_case_wait}s "
                "before next case..."
            )

            time.sleep(
                args.inter_case_wait
            )

    order = {
        case[
            "case_id"
        ]: index
        for index, case
        in enumerate(
            dataset[
                "cases"
            ]
        )
    }

    all_records = sorted(
        checkpoint.get(
            "records",
            {},
        ).values(),
        key=lambda item:
            order.get(
                item[
                    "case_id"
                ],
                10_000,
            ),
    )

    successful_records = [
        item
        for item
        in all_records
        if "error" not in item
    ]

    invalid_records = [
        item
        for item
        in all_records
        if "error" in item
    ]

    final_payload = {
        "evaluation_name":
            dataset[
                "benchmark_name"
            ],

        "dataset_version":
            dataset[
                "version"
            ],

        "runner_version":
            RUNNER_VERSION,

        "run_configuration": {
            "review_mode":
                "cross_judge",

            "judges": [
                "Step-3.5-Flash",
                "Qwen3.5-35B-A3B",
            ],

            "revision_model":
                args.revision_model,

            "provider_failure_policy":
                (
                    "Judge/provider failures invalidate the case, "
                    "trigger retry, and are never counted as "
                    "HUMAN_REVIEW_REQUIRED evaluation outcomes."
                ),

            "optional_optimization_policy":
                (
                    "Simulated only for advisory_only cases "
                    "when advisories exist."
                ),
        },

        "automated_summary":
            automated_summary(
                successful_records
            ),

        "invalid_case_ids": [
            item[
                "case_id"
            ]
            for item
            in invalid_records
        ],

        "records":
            all_records,
    }

    save_json(
        FINAL_FILE,
        final_payload,
    )

    blind_packet = (
        create_blind_packet(
            dataset,
            successful_records,
        )
    )

    save_json(
        BLIND_PACKET_FILE,
        blind_packet,
    )

    write_ratings_template(
        RATINGS_TEMPLATE_FILE,
        successful_records,
    )

    print(
        "\n"
        + "=" * 78
    )

    if invalid_records:

        print(
            "RUN INCOMPLETE — "
            "INVALID CASES MUST BE RERUN"
        )

    else:

        print(
            "RUN COMPLETE — "
            "ALL SAVED CASES ARE VALID TWO-JUDGE RUNS"
        )

    print(
        "=" * 78
    )

    print(
        f"Valid records: "
        f"{len(successful_records)}"
    )

    print(
        f"Invalid records: "
        f"{len(invalid_records)}"
    )

    if invalid_records:

        print(
            "Invalid case IDs: "
            + ", ".join(
                item[
                    "case_id"
                ]
                for item
                in invalid_records
            )
        )

    print(
        f"Final results: "
        f"{FINAL_FILE}"
    )

    print(
        f"Blind packet: "
        f"{BLIND_PACKET_FILE}"
    )

    print(
        f"Ratings template: "
        f"{RATINGS_TEMPLATE_FILE}"
    )

    print(
        "\nIf a provider/judge call fails, rerun the same "
        "command. Valid cases are skipped; invalid cases "
        "are retried automatically."
    )


if __name__ == "__main__":
    main()