import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from src.evaluator import evaluate_content
from src.llm_client import MODELS


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "creator_draft_stress_cases.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CASE_RESULTS_FILE = (
    RESULTS_DIR
    / "creator_stress_baseline_results.csv"
)

JUDGE_RESULTS_FILE = (
    RESULTS_DIR
    / "creator_stress_baseline_judge_results.csv"
)

METRICS_FILE = (
    RESULTS_DIR
    / "creator_stress_baseline_metrics.json"
)


# =========================================================
# Experiment Configuration
# =========================================================

JUDGE_MODELS = [
    "step",
    "qwen",
]

# One normal attempt + one retry.
JUDGE_MAX_ATTEMPTS = 2


# =========================================================
# Load + Validate Dataset
# =========================================================

def load_cases() -> list:
    """
    Load Benchmark B:

    data/creator_draft_stress_cases.json
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Stress dataset not found: {DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        cases = data

    elif (
        isinstance(data, dict)
        and "cases" in data
    ):
        cases = data["cases"]

    else:
        raise ValueError(
            "creator_draft_stress_cases.json must contain "
            "a list of cases or a {'cases': [...]} object."
        )

    if not cases:
        raise ValueError(
            "No stress cases were found."
        )

    return cases


def validate_case(
    case: dict,
    case_index: int,
):
    """
    Validate fields required by the baseline runner.
    """

    case_id = (
        case.get("case_id")
        or f"STRESS_{case_index:03d}"
    )

    required_text_fields = [
        "brand_info",
        "product_info",
        "campaign_brief",
        "creator_draft",
    ]

    for field in required_text_fields:
        if field not in case:
            raise ValueError(
                f"{case_id} is missing required field: "
                f"{field}"
            )

        if not str(
            case[field]
        ).strip():
            raise ValueError(
                f"{case_id} has empty required field: "
                f"{field}"
            )

    if "gold" not in case:
        raise ValueError(
            f"{case_id} is missing required field: gold"
        )

    if not isinstance(
        case["gold"],
        dict,
    ):
        raise ValueError(
            f"{case_id}.gold must be an object."
        )


# =========================================================
# Context Composition
# =========================================================

def format_list_section(
    title: str,
    values,
) -> str:
    """
    Format a list into a stable text section.
    """

    values = values or []

    if not values:
        return f"{title}:\n- None specified"

    lines = [f"{title}:"]

    for value in values:
        lines.append(f"- {value}")

    return "\n".join(lines)


def format_creator_profile(
    creator_profile,
) -> str:
    """
    Format Creator Profile without changing
    evaluator.py's public interface.
    """

    if not isinstance(creator_profile, dict):
        creator_profile = {}

    mapping = [
        (
            "Creator Category",
            creator_profile.get(
                "creator_category",
                "",
            ),
        ),
        (
            "Creator Audience",
            creator_profile.get(
                "audience",
                "",
            ),
        ),
        (
            "Creator Style",
            creator_profile.get(
                "creator_style",
                "",
            ),
        ),
        (
            "Content Characteristics",
            creator_profile.get(
                "content_characteristics",
                "",
            ),
        ),
    ]

    lines = [
        "CREATOR PROFILE:"
    ]

    for label, value in mapping:

        value = str(
            value
            or ""
        ).strip()

        if value:
            lines.append(
                f"- {label}: {value}"
            )

    if len(lines) == 1:
        lines.append(
            "- No creator profile supplied"
        )

    return "\n".join(
        lines
    )


def build_evaluator_contexts(
    case: dict,
) -> tuple[str, str, str]:
    """
    Adapt Product Definition v2 fields to the
    frozen evaluator.py interface.

    IMPORTANT:
    We do NOT change evaluator.py for this baseline.

    Existing evaluator accepts only:
    - brand_info
    - campaign_brief
    - generated_content
    - policy_context

    Therefore:
    - Product Information is appended to Brand Information.
    - Creator / platform / content-type / campaign requirement
      context is appended to Campaign Brief.
    - Explicit hard policy remains Additional Policy Context.
    """

    brand_info = str(
        case.get(
            "brand_info",
            "",
        )
    ).strip()

    product_info = str(
        case.get(
            "product_info",
            "",
        )
    ).strip()

    campaign_brief = str(
        case.get(
            "campaign_brief",
            "",
        )
    ).strip()

    platform = str(
        case.get(
            "platform",
            "",
        )
    ).strip()

    content_type = str(
        case.get(
            "content_type",
            "",
        )
    ).strip()

    creator_profile = (
        case.get(
            "creator_profile",
            {},
        )
        or {}
    )

    must_mention = (
        case.get(
            "must_mention",
            [],
        )
        or []
    )

    must_avoid = (
        case.get(
            "must_avoid",
            [],
        )
        or []
    )

    policy_context = str(
        case.get(
            "additional_policy_context",
            case.get(
                "policy_context",
                "",
            ),
        )
        or ""
    ).strip()

    brand_context = (
        f"{brand_info}\n\n"
        "VERIFIED PRODUCT INFORMATION:\n"
        f"{product_info}"
    ).strip()

    campaign_sections = [
        "ORIGINAL CAMPAIGN BRIEF:",
        campaign_brief,
        "",
        (
            "PLATFORM:\n"
            f"{platform or 'Not specified'}"
        ),
        "",
        (
            "CONTENT TYPE:\n"
            f"{content_type or 'Not specified'}"
        ),
        "",
        format_creator_profile(
            creator_profile
        ),
        "",
        format_list_section(
            "CAMPAIGN MUST MENTION",
            must_mention,
        ),
        "",
        format_list_section(
            "CAMPAIGN MUST AVOID",
            must_avoid,
        ),
    ]

    campaign_context = (
        "\n".join(
            campaign_sections
        ).strip()
    )

    return (
        brand_context,
        campaign_context,
        policy_context,
    )


# =========================================================
# Evaluation Helpers
# =========================================================

def get_blocking_count(
    evaluation: dict,
) -> int:

    if (
        "blocking_compliance_issue_count"
        in evaluation
    ):

        return int(
            evaluation[
                "blocking_compliance_issue_count"
            ]
        )

    return len(
        evaluation.get(
            "compliance_findings",
            [],
        )
    )


def get_heuristic_score(
    evaluation: dict,
):

    if (
        "heuristic_composite_score"
        in evaluation
    ):
        return evaluation[
            "heuristic_composite_score"
        ]

    if (
        "overall_score"
        in evaluation
    ):
        return evaluation[
            "overall_score"
        ]

    return None


def normalize_match_text(
    text: str,
) -> str:

    value = unicodedata.normalize(
        "NFKC",
        str(
            text
            or ""
        ),
    ).casefold()

    value = re.sub(
        r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+",
        "",
        value,
        flags=re.UNICODE,
    )

    return value


def evidence_matches(
    gold_evidence: str,
    predicted_evidence: str,
) -> bool:

    gold = normalize_match_text(
        gold_evidence
    )

    predicted = normalize_match_text(
        predicted_evidence
    )

    if (
        not gold
        or not predicted
    ):
        return False

    if min(
        len(gold),
        len(predicted),
    ) < 4:
        return False

    return (
        gold in predicted
        or predicted in gold
    )


def count_gold_findings_detected(
    gold_blocking_issues: list,
    predicted_findings: list,
) -> tuple[int, list]:

    detected_ids = []

    for gold_issue in (
        gold_blocking_issues
        or []
    ):

        gold_evidence = str(
            gold_issue.get(
                "evidence",
                "",
            )
        ).strip()

        matched = any(
            evidence_matches(
                gold_evidence,
                str(
                    finding.get(
                        "evidence",
                        "",
                    )
                ),
            )
            for finding
            in (
                predicted_findings
                or []
            )
        )

        if matched:
            detected_ids.append(
                gold_issue.get(
                    "issue_id",
                    "",
                )
            )

    return (
        len(
            detected_ids
        ),
        detected_ids,
    )


def detect_legacy_requirement_signal(
    evaluation: dict,
    required_missing: list,
) -> bool:
    """
    Diagnostic only.

    The frozen evaluator has no structured
    requirement_findings field.
    """

    if not required_missing:
        return False

    legacy_text = normalize_match_text(
        json.dumps(
            {
                "advisory_findings":
                    evaluation.get(
                        "advisory_findings",
                        [],
                    ),

                "review_notes":
                    evaluation.get(
                        "review_notes",
                        [],
                    ),
            },
            ensure_ascii=False,
        )
    )

    for requirement in required_missing:

        missing_item = (
            normalize_match_text(
                requirement.get(
                    "missing_item",
                    "",
                )
            )
        )

        if (
            missing_item
            and missing_item
            in legacy_text
        ):
            return True

    return False


# =========================================================
# Save Helpers
# =========================================================

def save_judge_results(
    judge_rows: list,
):

    if judge_rows:

        pd.DataFrame(
            judge_rows
        ).to_csv(
            JUDGE_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


def save_case_results(
    case_rows: list,
):

    if case_rows:

        pd.DataFrame(
            case_rows
        ).to_csv(
            CASE_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


# =========================================================
# Judge Execution
# =========================================================

def run_judge(
    case: dict,
    case_index: int,
    judge_model: str,
) -> tuple[dict, dict]:

    case_id = (
        case.get(
            "case_id"
        )
        or f"STRESS_{case_index:03d}"
    )

    scenario_name = str(
        case.get(
            "scenario_name",
            "",
        )
    )

    scenario_type = str(
        case.get(
            "scenario_type",
            "",
        )
    )

    creator_draft = str(
        case.get(
            "creator_draft",
            "",
        )
    ).strip()

    gold = (
        case.get(
            "gold",
            {},
        )
        or {}
    )

    (
        brand_context,
        campaign_context,
        policy_context,
    ) = build_evaluator_contexts(
        case
    )

    evaluation = None
    final_error = None
    successful_attempt = None
    total_latency = 0.0

    for attempt in range(
        1,
        JUDGE_MAX_ATTEMPTS + 1,
    ):

        print(
            f"      Judge={judge_model} "
            f"Attempt={attempt}/"
            f"{JUDGE_MAX_ATTEMPTS}"
        )

        start_time = (
            time.perf_counter()
        )

        try:

            evaluation = (
                evaluate_content(
                    brand_info=(
                        brand_context
                    ),

                    campaign_brief=(
                        campaign_context
                    ),

                    generated_content=(
                        creator_draft
                    ),

                    policy_context=(
                        policy_context
                    ),

                    judge_model_key=(
                        judge_model
                    ),
                )
            )

            attempt_latency = (
                time.perf_counter()
                - start_time
            )

            total_latency += (
                attempt_latency
            )

            successful_attempt = (
                attempt
            )

            final_error = None

            print(
                f"      ✅ "
                f"Heuristic="
                f"{get_heuristic_score(evaluation)}"
                f" | Risk="
                f"{evaluation.get('unsupported_claim_risk')}"
                f" | Blocking="
                f"{get_blocking_count(evaluation)}"
                f" | Latency="
                f"{attempt_latency:.2f}s"
            )

            break

        except Exception as error:

            attempt_latency = (
                time.perf_counter()
                - start_time
            )

            total_latency += (
                attempt_latency
            )

            final_error = str(
                error
            )

            print(
                f"      ⚠️ Failed after "
                f"{attempt_latency:.2f}s: "
                f"{error}"
            )

            if (
                attempt
                < JUDGE_MAX_ATTEMPTS
            ):

                print(
                    "      Retrying..."
                )

    total_latency = round(
        total_latency,
        2,
    )

    gold_blocking_issues = (
        gold.get(
            "blocking_issues",
            [],
        )
        or []
    )

    required_missing = (
        gold.get(
            "required_missing",
            [],
        )
        or []
    )

    # =====================================================
    # Judge Success
    # =====================================================

    if evaluation is not None:

        predicted_findings = (
            evaluation.get(
                "compliance_findings",
                [],
            )
            or []
        )

        (
            detected_gold_finding_count,
            detected_gold_issue_ids,
        ) = count_gold_findings_detected(
            gold_blocking_issues,
            predicted_findings,
        )

        legacy_requirement_signal = (
            detect_legacy_requirement_signal(
                evaluation,
                required_missing,
            )
        )

        blocking_count = (
            get_blocking_count(
                evaluation
            )
        )

        judge_result = {
            "evaluation":
                evaluation,

            "latency":
                total_latency,

            "attempts":
                successful_attempt,

            "error":
                None,

            "blocking_flag":
                blocking_count > 0,

            "detected_gold_finding_count":
                detected_gold_finding_count,

            "detected_gold_issue_ids":
                detected_gold_issue_ids,

            "legacy_requirement_signal":
                legacy_requirement_signal,
        }

        judge_row = {
            "case_id":
                case_id,

            "scenario_name":
                scenario_name,

            "scenario_type":
                scenario_type,

            "judge_model":
                judge_model,

            "judge_model_id":
                MODELS[
                    judge_model
                ],

            "status":
                "SUCCESS",

            "attempts":
                successful_attempt,

            "evaluation_latency":
                total_latency,

            "gold_blocking_expected":
                bool(
                    gold.get(
                        "blocking_expected",
                        False,
                    )
                ),

            "gold_blocking_issue_count":
                int(
                    gold.get(
                        "expected_blocking_issue_count",
                        0,
                    )
                    or 0
                ),

            "gold_required_missing_count":
                len(
                    required_missing
                ),

            "negative_control":
                bool(
                    gold.get(
                        "negative_control",
                        False,
                    )
                ),

            "predicted_blocking_flag":
                blocking_count > 0,

            "predicted_blocking_count":
                blocking_count,

            "detected_gold_finding_count":
                detected_gold_finding_count,

            "detected_gold_issue_ids":
                json.dumps(
                    detected_gold_issue_ids,
                    ensure_ascii=False,
                ),

            "legacy_requirement_signal":
                legacy_requirement_signal,

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
                get_heuristic_score(
                    evaluation
                ),

            "compliance_status":
                evaluation.get(
                    "compliance_status",
                    "",
                ),

            "compliance_findings":
                json.dumps(
                    evaluation.get(
                        "compliance_findings",
                        [],
                    ),
                    ensure_ascii=False,
                ),

            "advisory_findings":
                json.dumps(
                    evaluation.get(
                        "advisory_findings",
                        [],
                    ),
                    ensure_ascii=False,
                ),

            "review_notes":
                json.dumps(
                    evaluation.get(
                        "review_notes",
                        [],
                    ),
                    ensure_ascii=False,
                ),

            "error":
                "",
        }

    # =====================================================
    # Judge Failure
    # =====================================================

    else:

        judge_result = {
            "evaluation":
                None,

            "latency":
                total_latency,

            "attempts":
                JUDGE_MAX_ATTEMPTS,

            "error":
                final_error,

            "blocking_flag":
                None,

            "detected_gold_finding_count":
                0,

            "detected_gold_issue_ids":
                [],

            "legacy_requirement_signal":
                False,
        }

        judge_row = {
            "case_id":
                case_id,

            "scenario_name":
                scenario_name,

            "scenario_type":
                scenario_type,

            "judge_model":
                judge_model,

            "judge_model_id":
                MODELS[
                    judge_model
                ],

            "status":
                "FAILED",

            "attempts":
                JUDGE_MAX_ATTEMPTS,

            "evaluation_latency":
                total_latency,

            "gold_blocking_expected":
                bool(
                    gold.get(
                        "blocking_expected",
                        False,
                    )
                ),

            "gold_blocking_issue_count":
                int(
                    gold.get(
                        "expected_blocking_issue_count",
                        0,
                    )
                    or 0
                ),

            "gold_required_missing_count":
                len(
                    required_missing
                ),

            "negative_control":
                bool(
                    gold.get(
                        "negative_control",
                        False,
                    )
                ),

            "predicted_blocking_flag":
                None,

            "predicted_blocking_count":
                None,

            "detected_gold_finding_count":
                0,

            "detected_gold_issue_ids":
                "[]",

            "legacy_requirement_signal":
                False,

            "brand_alignment":
                None,

            "tone_match":
                None,

            "selling_point_coverage":
                None,

            "factual_consistency":
                None,

            "unsupported_claim_risk":
                None,

            "heuristic_composite_score":
                None,

            "compliance_status":
                "",

            "compliance_findings":
                "",

            "advisory_findings":
                "",

            "review_notes":
                "",

            "error":
                final_error
                or "",
        }

    return (
        judge_result,
        judge_row,
    )


# =========================================================
# Cross-Judge Aggregation
# =========================================================

def aggregate_case_judges(
    judge_results: dict,
) -> dict:

    valid = {
        judge_model:
            result

        for (
            judge_model,
            result
        )
        in judge_results.items()

        if result.get(
            "evaluation"
        ) is not None
    }

    if len(valid) != len(
        JUDGE_MODELS
    ):

        return {
            "panel_complete":
                False,

            "blocking_agreement":
                None,

            "compliance_decision":
                "INCOMPLETE_PANEL",

            "avg_heuristic_score":
                None,

            "avg_claim_risk":
                None,

            "judge_score_gap":
                None,
        }

    blocking_flags = [
        bool(
            valid[
                model
            ][
                "blocking_flag"
            ]
        )

        for model
        in JUDGE_MODELS
    ]

    heuristic_scores = [
        get_heuristic_score(
            valid[
                model
            ][
                "evaluation"
            ]
        )

        for model
        in JUDGE_MODELS
    ]

    claim_risks = [
        valid[
            model
        ][
            "evaluation"
        ].get(
            "unsupported_claim_risk"
        )

        for model
        in JUDGE_MODELS
    ]

    if all(
        blocking_flags
    ):

        decision = (
            "CONSENSUS_BLOCKING"
        )

    elif not any(
        blocking_flags
    ):

        decision = (
            "CONSENSUS_NO_BLOCKING"
        )

    else:

        decision = (
            "JUDGE_DISAGREEMENT"
        )

    return {
        "panel_complete":
            True,

        "blocking_agreement":
            (
                len(
                    set(
                        blocking_flags
                    )
                )
                == 1
            ),

        "compliance_decision":
            decision,

        "avg_heuristic_score":
            round(
                sum(
                    heuristic_scores
                )
                / len(
                    heuristic_scores
                ),
                2,
            ),

        "avg_claim_risk":
            round(
                sum(
                    claim_risks
                )
                / len(
                    claim_risks
                ),
                2,
            ),

        "judge_score_gap":
            round(
                abs(
                    heuristic_scores[0]
                    - heuristic_scores[1]
                ),
                2,
            ),
    }


def classify_case_outcome(
    gold_blocking_expected: bool,
    compliance_decision: str,
) -> str:

    if (
        compliance_decision
        == "INCOMPLETE_PANEL"
    ):
        return "INCOMPLETE"

    if (
        compliance_decision
        == "JUDGE_DISAGREEMENT"
    ):
        return "HUMAN_REVIEW"

    predicted_blocking = (
        compliance_decision
        == "CONSENSUS_BLOCKING"
    )

    if (
        gold_blocking_expected
        and predicted_blocking
    ):
        return "TRUE_POSITIVE"

    if (
        not gold_blocking_expected
        and not predicted_blocking
    ):
        return "TRUE_NEGATIVE"

    if (
        gold_blocking_expected
        and not predicted_blocking
    ):
        return "FALSE_NEGATIVE"

    return "FALSE_POSITIVE"


# =========================================================
# Case Evaluation
# =========================================================

def run_case(
    case: dict,
    case_index: int,
    judge_rows: list,
) -> dict:

    validate_case(
        case,
        case_index,
    )

    case_id = (
        case.get(
            "case_id"
        )
        or f"STRESS_{case_index:03d}"
    )

    scenario_name = str(
        case.get(
            "scenario_name",
            "",
        )
    )

    scenario_type = str(
        case.get(
            "scenario_type",
            "",
        )
    )

    gold = (
        case.get(
            "gold",
            {},
        )
        or {}
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"[{case_index}] "
        f"{case_id} | "
        f"{scenario_name}"
    )

    print(
        f"Scenario type: "
        f"{scenario_type}"
    )

    print(
        "=" * 80
    )

    judge_results = {}

    for judge_model in JUDGE_MODELS:

        (
            judge_result,
            judge_row,
        ) = run_judge(
            case=case,

            case_index=case_index,

            judge_model=judge_model,
        )

        judge_results[
            judge_model
        ] = judge_result

        judge_rows.append(
            judge_row
        )

        save_judge_results(
            judge_rows
        )

    aggregation = (
        aggregate_case_judges(
            judge_results
        )
    )

    gold_blocking_expected = (
        bool(
            gold.get(
                "blocking_expected",
                False,
            )
        )
    )

    compliance_decision = (
        aggregation[
            "compliance_decision"
        ]
    )

    baseline_outcome = (
        classify_case_outcome(
            gold_blocking_expected,
            compliance_decision,
        )
    )

    return {
        "case_id":
            case_id,

        "scenario_name":
            scenario_name,

        "scenario_type":
            scenario_type,

        "platform":
            case.get(
                "platform",
                "",
            ),

        "content_type":
            case.get(
                "content_type",
                "",
            ),

        "gold_blocking_expected":
            gold_blocking_expected,

        "gold_blocking_issue_count":
            int(
                gold.get(
                    "expected_blocking_issue_count",
                    0,
                )
                or 0
            ),

        "gold_required_missing_count":
            len(
                gold.get(
                    "required_missing",
                    [],
                )
                or []
            ),

        "gold_advisory_expected":
            bool(
                gold.get(
                    "advisory_expected",
                    False,
                )
            ),

        "negative_control":
            bool(
                gold.get(
                    "negative_control",
                    False,
                )
            ),

        "desired_product_action":
            gold.get(
                "desired_product_action",
                "",
            ),

        "panel_complete":
            aggregation[
                "panel_complete"
            ],

        "blocking_agreement":
            aggregation[
                "blocking_agreement"
            ],

        "compliance_decision":
            compliance_decision,

        "baseline_outcome":
            baseline_outcome,

        "avg_heuristic_score":
            aggregation[
                "avg_heuristic_score"
            ],

        "avg_claim_risk":
            aggregation[
                "avg_claim_risk"
            ],

        "judge_score_gap":
            aggregation[
                "judge_score_gap"
            ],

        "step_blocking_flag":
            judge_results[
                "step"
            ].get(
                "blocking_flag"
            ),

        "qwen_blocking_flag":
            judge_results[
                "qwen"
            ].get(
                "blocking_flag"
            ),

        "step_detected_gold_finding_count":
            judge_results[
                "step"
            ].get(
                "detected_gold_finding_count",
                0,
            ),

        "qwen_detected_gold_finding_count":
            judge_results[
                "qwen"
            ].get(
                "detected_gold_finding_count",
                0,
            ),

        "step_legacy_requirement_signal":
            judge_results[
                "step"
            ].get(
                "legacy_requirement_signal",
                False,
            ),

        "qwen_legacy_requirement_signal":
            judge_results[
                "qwen"
            ].get(
                "legacy_requirement_signal",
                False,
            ),

        "creator_draft":
            case.get(
                "creator_draft",
                "",
            ),
    }


# =========================================================
# Metrics
# =========================================================

def safe_rate(
    numerator: int,
    denominator: int,
):

    if denominator == 0:
        return None

    return round(
        numerator
        / denominator,
        4,
    )


def calculate_metrics(
    cases: list,
    case_rows: list,
    judge_rows: list,
) -> dict:

    case_df = pd.DataFrame(
        case_rows
    )

    judge_df = pd.DataFrame(
        judge_rows
    )

    successful_judges = (
        judge_df[
            judge_df[
                "status"
            ]
            == "SUCCESS"
        ]
        .copy()
    )

    complete_cases = (
        case_df[
            case_df[
                "panel_complete"
            ]
            == True
        ]
        .copy()
    )

    gold_blocking_cases = (
        complete_cases[
            complete_cases[
                "gold_blocking_expected"
            ]
            == True
        ]
    )

    gold_nonblocking_cases = (
        complete_cases[
            complete_cases[
                "gold_blocking_expected"
            ]
            == False
        ]
    )

    consensus_blocking_on_gold = len(
        gold_blocking_cases[
            gold_blocking_cases[
                "compliance_decision"
            ]
            == "CONSENSUS_BLOCKING"
        ]
    )

    false_consensus_blocking = len(
        gold_nonblocking_cases[
            gold_nonblocking_cases[
                "compliance_decision"
            ]
            == "CONSENSUS_BLOCKING"
        ]
    )

    cross_judge_agreement_count = int(
        complete_cases[
            "blocking_agreement"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )

    disagreement_count = len(
        complete_cases[
            complete_cases[
                "compliance_decision"
            ]
            == "JUDGE_DISAGREEMENT"
        ]
    )

    negative_controls = (
        complete_cases[
            complete_cases[
                "negative_control"
            ]
            == True
        ]
    )

    negative_controls_clear = len(
        negative_controls[
            negative_controls[
                "compliance_decision"
            ]
            == "CONSENSUS_NO_BLOCKING"
        ]
    )

    advisory_only_cases = (
        complete_cases[
            complete_cases[
                "scenario_type"
            ]
            == "advisory_only"
        ]
    )

    advisory_only_clear = len(
        advisory_only_cases[
            advisory_only_cases[
                "compliance_decision"
            ]
            == "CONSENSUS_NO_BLOCKING"
        ]
    )

    individual_gold_blocking = (
        successful_judges[
            successful_judges[
                "gold_blocking_expected"
            ]
            == True
        ]
    )

    individual_gold_nonblocking = (
        successful_judges[
            successful_judges[
                "gold_blocking_expected"
            ]
            == False
        ]
    )

    individual_detected_blocking = int(
        individual_gold_blocking[
            "predicted_blocking_flag"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )

    individual_false_blocking = int(
        individual_gold_nonblocking[
            "predicted_blocking_flag"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )

    total_gold_findings_available = int(
        individual_gold_blocking[
            "gold_blocking_issue_count"
        ]
        .fillna(
            0
        )
        .sum()
    )

    total_gold_findings_detected = int(
        individual_gold_blocking[
            "detected_gold_finding_count"
        ]
        .fillna(
            0
        )
        .sum()
    )

    per_judge = {}

    for judge_model in JUDGE_MODELS:

        judge_data = (
            successful_judges[
                successful_judges[
                    "judge_model"
                ]
                == judge_model
            ]
        )

        judge_gold_blocking = (
            judge_data[
                judge_data[
                    "gold_blocking_expected"
                ]
                == True
            ]
        )

        judge_gold_nonblocking = (
            judge_data[
                judge_data[
                    "gold_blocking_expected"
                ]
                == False
            ]
        )

        judge_gold_findings = int(
            judge_gold_blocking[
                "gold_blocking_issue_count"
            ]
            .fillna(
                0
            )
            .sum()
        )

        judge_detected_findings = int(
            judge_gold_blocking[
                "detected_gold_finding_count"
            ]
            .fillna(
                0
            )
            .sum()
        )

        per_judge[
            judge_model
        ] = {
            "successful_calls":
                int(
                    len(
                        judge_data
                    )
                ),

            "blocking_case_recall":
                safe_rate(
                    int(
                        judge_gold_blocking[
                            "predicted_blocking_flag"
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        )
                        .sum()
                    ),
                    int(
                        len(
                            judge_gold_blocking
                        )
                    ),
                ),

            "false_blocking_rate":
                safe_rate(
                    int(
                        judge_gold_nonblocking[
                            "predicted_blocking_flag"
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        )
                        .sum()
                    ),
                    int(
                        len(
                            judge_gold_nonblocking
                        )
                    ),
                ),

            "blocking_finding_recall":
                safe_rate(
                    judge_detected_findings,
                    judge_gold_findings,
                ),

            "avg_evaluation_latency_seconds":
                (
                    round(
                        float(
                            judge_data[
                                "evaluation_latency"
                            ].mean()
                        ),
                        2,
                    )

                    if not judge_data.empty

                    else None
                ),
        }

    requirement_cases = (
        successful_judges[
            successful_judges[
                "gold_required_missing_count"
            ]
            > 0
        ]
    )

    legacy_requirement_signal_count = int(
        requirement_cases[
            "legacy_requirement_signal"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )

    return {
        "benchmark":
            "Creator Draft Stress Baseline",

        "baseline_system":
            (
                "Frozen evaluator.py before structured "
                "Requirement Findings"
            ),

        "total_cases":
            int(
                len(
                    cases
                )
            ),

        "completed_case_panels":
            int(
                len(
                    complete_cases
                )
            ),

        "total_expected_judge_calls":
            int(
                len(
                    cases
                )
                * len(
                    JUDGE_MODELS
                )
            ),

        "successful_judge_calls":
            int(
                len(
                    successful_judges
                )
            ),

        "gold_blocking_cases_in_complete_panels":
            int(
                len(
                    gold_blocking_cases
                )
            ),

        "gold_nonblocking_cases_in_complete_panels":
            int(
                len(
                    gold_nonblocking_cases
                )
            ),

        "consensus_blocking_recall_for_auto_fix":
            safe_rate(
                consensus_blocking_on_gold,
                len(
                    gold_blocking_cases
                ),
            ),

        "false_consensus_blocking_rate":
            safe_rate(
                false_consensus_blocking,
                len(
                    gold_nonblocking_cases
                ),
            ),

        "cross_judge_blocking_agreement_rate":
            safe_rate(
                cross_judge_agreement_count,
                len(
                    complete_cases
                ),
            ),

        "human_review_escalation_rate":
            safe_rate(
                disagreement_count,
                len(
                    complete_cases
                ),
            ),

        "negative_control_clear_rate":
            safe_rate(
                negative_controls_clear,
                len(
                    negative_controls
                ),
            ),

        "advisory_only_clear_rate":
            safe_rate(
                advisory_only_clear,
                len(
                    advisory_only_cases
                ),
            ),

        "individual_judge_blocking_case_recall":
            safe_rate(
                individual_detected_blocking,
                len(
                    individual_gold_blocking
                ),
            ),

        "individual_judge_false_blocking_rate":
            safe_rate(
                individual_false_blocking,
                len(
                    individual_gold_nonblocking
                ),
            ),

        "individual_judge_blocking_finding_recall":
            safe_rate(
                total_gold_findings_detected,
                total_gold_findings_available,
            ),

        "requirement_coverage_baseline": {
            "structured_requirement_findings_supported":
                False,

            "formal_requirement_detection_rate":
                None,

            "reason":
                (
                    "The frozen evaluator has no "
                    "requirement_findings output schema. "
                    "Missing Must-Mention items may only "
                    "surface indirectly as Advisory or "
                    "Selling Point Coverage."
                ),

            "legacy_advisory_signal_calls":
                legacy_requirement_signal_count,

            "legacy_advisory_signal_total_successful_calls":
                int(
                    len(
                        requirement_cases
                    )
                ),

            "note":
                (
                    "Legacy advisory signal is diagnostic "
                    "only and must not be presented as "
                    "formal Requirement Accuracy."
                ),
        },

        "per_judge":
            per_judge,
    }


def save_metrics(
    metrics: dict,
):

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# Console Summary
# =========================================================

def format_rate(
    value,
) -> str:

    if value is None:
        return "N/A"

    return f"{value:.1%}"


def print_summary(
    metrics: dict,
):

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "CREATOR DRAFT STRESS BASELINE SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\nCompleted panels: "
        f"{metrics['completed_case_panels']}/"
        f"{metrics['total_cases']}"
    )

    print(
        "Successful Judge calls: "
        f"{metrics['successful_judge_calls']}/"
        f"{metrics['total_expected_judge_calls']}"
    )

    print(
        "\nConsensus Blocking Recall "
        "(auto-fix eligibility): "
        f"{format_rate(
            metrics[
                'consensus_blocking_recall_for_auto_fix'
            ]
        )}"
    )

    print(
        "False Consensus Blocking Rate: "
        f"{format_rate(
            metrics[
                'false_consensus_blocking_rate'
            ]
        )}"
    )

    print(
        "Cross-Judge Blocking Agreement: "
        f"{format_rate(
            metrics[
                'cross_judge_blocking_agreement_rate'
            ]
        )}"
    )

    print(
        "Human Review Escalation Rate: "
        f"{format_rate(
            metrics[
                'human_review_escalation_rate'
            ]
        )}"
    )

    print(
        "Negative Control Clear Rate: "
        f"{format_rate(
            metrics[
                'negative_control_clear_rate'
            ]
        )}"
    )

    print(
        "Advisory-only Clear Rate: "
        f"{format_rate(
            metrics[
                'advisory_only_clear_rate'
            ]
        )}"
    )

    print(
        "\nIndividual Judge Blocking "
        "Case Recall: "
        f"{format_rate(
            metrics[
                'individual_judge_blocking_case_recall'
            ]
        )}"
    )

    print(
        "Individual Judge False Blocking Rate: "
        f"{format_rate(
            metrics[
                'individual_judge_false_blocking_rate'
            ]
        )}"
    )

    print(
        "Individual Judge Blocking "
        "Finding Recall: "
        f"{format_rate(
            metrics[
                'individual_judge_blocking_finding_recall'
            ]
        )}"
    )

    req = (
        metrics[
            "requirement_coverage_baseline"
        ]
    )

    print(
        "\nRequirement Coverage:"
    )

    print(
        "  Structured findings supported: "
        "False"
    )

    print(
        "  Formal detection rate: "
        "N/A (architectural baseline gap)"
    )

    print(
        "  Legacy advisory signals: "
        f"{req['legacy_advisory_signal_calls']}/"
        f"{req[
            'legacy_advisory_signal_total_successful_calls'
        ]}"
    )

    print(
        "\nPer Judge:"
    )

    for (
        judge_model,
        judge_metrics,
    ) in metrics[
        "per_judge"
    ].items():

        print(
            f"  {judge_model}: "
            f"case recall="
            f"{format_rate(
                judge_metrics[
                    'blocking_case_recall'
                ]
            )}, "
            f"false blocking="
            f"{format_rate(
                judge_metrics[
                    'false_blocking_rate'
                ]
            )}, "
            f"finding recall="
            f"{format_rate(
                judge_metrics[
                    'blocking_finding_recall'
                ]
            )}, "
            f"avg latency="
            f"{judge_metrics[
                'avg_evaluation_latency_seconds'
            ]}s"
        )

    print(
        "\nResults:"
    )

    print(
        f"  {CASE_RESULTS_FILE}"
    )

    print(
        f"  {JUDGE_RESULTS_FILE}"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print(
        "\nInterpretation rule:"
    )

    print(
        "  Numerical quality scores remain "
        "diagnostic only."
    )

    print(
        "  Blocking is evaluated against "
        "Gold hard constraints."
    )

    print(
        "  Requirement Coverage has no formal "
        "baseline accuracy because the current "
        "evaluator has no structured "
        "requirement_findings layer."
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    cases = (
        load_cases()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GrowthPilot Creator Draft Stress Baseline"
    )

    print(
        "=" * 80
    )

    print(
        f"\nCases: "
        f"{len(cases)}"
    )

    print(
        f"Judges: "
        f"{JUDGE_MODELS}"
    )

    print(
        f"Judge Max Attempts: "
        f"{JUDGE_MAX_ATTEMPTS}"
    )

    print(
        "Generator: NOT USED "
        "(creator draft is supplied)"
    )

    print(
        "Reviser: NOT USED "
        "(baseline evaluates current reviewer only)"
    )

    print(
        "Numerical Quality Threshold: None"
    )

    print(
        "\nContext adaptation:"
    )

    print(
        "  Product Information "
        "-> Brand Information context"
    )

    print(
        "  Creator/Profile/Platform/Requirements "
        "-> Campaign Brief context"
    )

    print(
        "  Additional Policy Context "
        "-> Policy Context"
    )

    case_rows = []

    judge_rows = []

    for (
        case_index,
        case,
    ) in enumerate(
        cases,
        start=1,
    ):

        try:

            row = (
                run_case(
                    case=case,
                    case_index=case_index,
                    judge_rows=judge_rows,
                )
            )

        except Exception as error:

            case_id = (
                case.get(
                    "case_id"
                )
                or f"STRESS_{case_index:03d}"
            )

            print(
                f"\n❌ Unexpected failure "
                f"for {case_id}: "
                f"{error}"
            )

            row = {
                "case_id":
                    case_id,

                "scenario_name":
                    case.get(
                        "scenario_name",
                        "",
                    ),

                "scenario_type":
                    case.get(
                        "scenario_type",
                        "",
                    ),

                "panel_complete":
                    False,

                "compliance_decision":
                    "FAILED_UNEXPECTED",

                "baseline_outcome":
                    "INCOMPLETE",

                "error":
                    str(
                        error
                    ),
            }

        case_rows.append(
            row
        )

        save_case_results(
            case_rows
        )

    metrics = (
        calculate_metrics(
            cases=cases,
            case_rows=case_rows,
            judge_rows=judge_rows,
        )
    )

    save_metrics(
        metrics
    )

    print_summary(
        metrics
    )