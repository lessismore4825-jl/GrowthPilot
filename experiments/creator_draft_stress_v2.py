import json
import statistics
import time
from pathlib import Path

import pandas as pd

from experiments.creator_draft_stress_baseline import (
    build_evaluator_contexts,
    count_gold_findings_detected,
    load_cases,
    normalize_match_text,
    safe_rate,
    validate_case,
)
from src.evaluator import evaluate_content, normalize_requirements
from src.llm_client import MODELS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CASE_RESULTS_FILE = RESULTS_DIR / "creator_stress_v2_results.csv"
JUDGE_RESULTS_FILE = RESULTS_DIR / "creator_stress_v2_judge_results.csv"
METRICS_FILE = RESULTS_DIR / "creator_stress_v2_metrics.json"
BASELINE_METRICS_FILE = RESULTS_DIR / "creator_stress_baseline_metrics.json"

ARCHITECTURE_VERSION = "v2.2"
JUDGE_MODELS = ["step", "qwen"]
JUDGE_MAX_ATTEMPTS = 2
CREATOR_SEMANTICS_CASE_IDS = {"STRESS_003", "STRESS_008"}

AUTHENTICITY_REGRESSION_PHRASES = [
    "invented personal experience",
    "fabricated personal experience",
    "fake personal experience",
    "unsupported personal experience",
    "unsubstantiated personal endorsement",
    "invented creator experience",
    "fabricated creator experience",
]

COPY_LIKE_MARKERS = [
    "for example",
    "e.g.",
    "e.g.,",
    "say:",
    "write:",
    "replace with:",
    "例如",
    "比如",
    "改成：",
    "写成：",
]


def get_requirements(case):
    if "structured_requirements" in case:
        return case.get("structured_requirements") or []

    return case.get("must_mention", []) or []


def blocking_count(evaluation):
    return int(
        evaluation.get(
            "blocking_compliance_issue_count",
            len(
                evaluation.get(
                    "compliance_findings",
                    [],
                )
                or []
            ),
        )
    )


def requirement_count(evaluation):
    if "requirement_finding_count" in evaluation:
        return int(
            evaluation[
                "requirement_finding_count"
            ]
        )

    if "requirement_missing_count" in evaluation:
        return int(
            evaluation[
                "requirement_missing_count"
            ]
        )

    return len(
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )


def heuristic_score(evaluation):
    return evaluation.get(
        "heuristic_composite_score",
        evaluation.get(
            "overall_score"
        ),
    )


def mean(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(
        sum(values) / len(values),
        2,
    )


def median(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(
        statistics.median(values),
        2,
    )


def gold_missing_text(item):
    if isinstance(
        item,
        dict,
    ):
        return str(
            item.get(
                "missing_item"
            )
            or item.get(
                "content"
            )
            or item.get(
                "requirement"
            )
            or item.get(
                "item"
            )
            or ""
        ).strip()

    return str(
        item or ""
    ).strip()


def requirement_match(
    gold_item,
    predicted,
):
    if isinstance(
        gold_item,
        dict,
    ):
        gold_id = str(
            gold_item.get(
                "requirement_id",
                "",
            )
            or ""
        ).strip()

        predicted_id = str(
            predicted.get(
                "requirement_id",
                "",
            )
            or ""
        ).strip()

        if (
            gold_id
            and predicted_id
            and gold_id == predicted_id
        ):
            return True

    gold_text = normalize_match_text(
        gold_missing_text(
            gold_item
        )
    )

    predicted_text = normalize_match_text(
        predicted.get(
            "requirement",
            "",
        )
    )

    if (
        not gold_text
        or not predicted_text
    ):
        return False

    return (
        gold_text == predicted_text
        or gold_text in predicted_text
        or predicted_text in gold_text
    )


def match_requirements(
    gold_items,
    predicted_items,
):
    gold_items = list(
        gold_items or []
    )

    predicted_items = list(
        predicted_items or []
    )

    matched_gold = set()
    matched_predicted = set()

    for gold_index, gold_item in enumerate(
        gold_items
    ):
        for predicted_index, predicted in enumerate(
            predicted_items
        ):
            if (
                predicted_index
                in matched_predicted
            ):
                continue

            if requirement_match(
                gold_item,
                predicted,
            ):
                matched_gold.add(
                    gold_index
                )

                matched_predicted.add(
                    predicted_index
                )

                break

    return {
        "detected_gold_count":
            len(
                matched_gold
            ),

        "matched_predicted_count":
            len(
                matched_predicted
            ),

        "extra_predicted_count":
            len(
                predicted_items
            )
            - len(
                matched_predicted
            ),

        "detected_gold_items":
            [
                gold_missing_text(
                    gold_items[index]
                )
                for index in sorted(
                    matched_gold
                )
            ],
    }


def requirement_signature(
    evaluation,
):
    values = []

    for finding in (
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    ):
        requirement_id = str(
            finding.get(
                "requirement_id",
                "",
            )
            or ""
        ).strip()

        if requirement_id:
            values.append(
                f"ID:{requirement_id}"
            )

        else:
            text = normalize_match_text(
                finding.get(
                    "requirement",
                    "",
                )
            )

            if text:
                values.append(
                    f"TEXT:{text}"
                )

    return tuple(
        sorted(
            set(values)
        )
    )


def creator_regression(
    evaluation,
):
    text = json.dumps(
        evaluation.get(
            "advisory_findings",
            [],
        )
        or [],
        ensure_ascii=False,
    ).casefold()

    return any(
        phrase.casefold()
        in text
        for phrase
        in AUTHENTICITY_REGRESSION_PHRASES
    )


def advisory_diagnostics(
    evaluation,
):
    counts = {
        "SUPPLIED_CONTEXT":
            0,

        "GENERAL_HEURISTIC":
            0,

        "SYSTEM_GROUNDING_REVIEW":
            0,

        "OTHER":
            0,
    }

    copy_like = 0

    for finding in (
        evaluation.get(
            "advisory_findings",
            [],
        )
        or []
    ):
        basis = str(
            finding.get(
                "basis_type",
                "",
            )
            or ""
        ).strip().upper()

        if basis in counts:
            counts[
                basis
            ] += 1
        else:
            counts[
                "OTHER"
            ] += 1

        suggestion = str(
            finding.get(
                "suggestion",
                "",
            )
            or ""
        ).casefold()

        if any(
            marker.casefold()
            in suggestion
            for marker
            in COPY_LIKE_MARKERS
        ):
            copy_like += 1

    relabels = sum(
        (
            "relabeled general_heuristic"
            in str(note).casefold()
        )
        for note in (
            evaluation.get(
                "review_notes",
                [],
            )
            or []
        )
    )

    return (
        counts,
        relabels,
        copy_like,
    )


def expected_action(
    gold,
):
    has_blocking = bool(
        gold.get(
            "blocking_expected",
            False,
        )
    )

    has_requirement = bool(
        gold.get(
            "required_missing",
            [],
        )
        or []
    )

    if (
        has_blocking
        and has_requirement
    ):
        return (
            "COMPLIANCE_AND_REQUIREMENT_ACTION"
        )

    if has_blocking:
        return (
            "COMPLIANCE_ACTION"
        )

    if has_requirement:
        return (
            "REQUIREMENT_ACTION"
        )

    return (
        "NO_MANDATORY_ACTION"
    )


def format_rate(value):
    if value is None:
        return "N/A"

    return f"{value:.1%}"


def save_csv(
    rows,
    path,
):
    if rows:
        pd.DataFrame(
            rows
        ).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )


def run_judge(
    case,
    case_index,
    judge_model,
):
    case_id = (
        case.get(
            "case_id"
        )
        or f"STRESS_{case_index:03d}"
    )

    gold = (
        case.get(
            "gold",
            {},
        )
        or {}
    )

    gold_blocking = (
        gold.get(
            "blocking_issues",
            [],
        )
        or []
    )

    gold_missing = (
        gold.get(
            "required_missing",
            [],
        )
        or []
    )

    requirements = get_requirements(
        case
    )

    normalized_requirements = (
        normalize_requirements(
            requirements
        )
    )

    (
        brand_context,
        campaign_context,
        policy_context,
    ) = build_evaluator_contexts(
        case
    )

    evaluation = None
    error = None
    successful_attempt = None

    total_latency = 0.0

    for attempt in range(
        1,
        JUDGE_MAX_ATTEMPTS + 1,
    ):
        print(
            f"      Judge={judge_model} "
            f"Attempt={attempt}/{JUDGE_MAX_ATTEMPTS}"
        )

        start = (
            time.perf_counter()
        )

        try:
            evaluation = evaluate_content(
                brand_info=brand_context,
                campaign_brief=campaign_context,
                generated_content=str(
                    case.get(
                        "creator_draft",
                        "",
                    )
                ).strip(),
                policy_context=policy_context,
                judge_model_key=judge_model,
                requirements=requirements,
                content_origin="creator_draft",
            )

            latency = (
                time.perf_counter()
                - start
            )

            total_latency += (
                latency
            )

            successful_attempt = (
                attempt
            )

            print(
                f"      ✅ "
                f"Heuristic={heuristic_score(evaluation)}"
                f" | Risk="
                f"{evaluation.get('unsupported_claim_risk')}"
                f" | Blocking="
                f"{blocking_count(evaluation)}"
                f" | Requirement="
                f"{requirement_count(evaluation)}"
                f" | Action="
                f"{evaluation.get('mandatory_action_status')}"
                f" | Latency="
                f"{latency:.2f}s"
            )

            break

        except Exception as exc:
            latency = (
                time.perf_counter()
                - start
            )

            total_latency += (
                latency
            )

            error = str(
                exc
            )

            print(
                f"      ⚠️ Failed after "
                f"{latency:.2f}s: {exc}"
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

    base = {
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

        "judge_model":
            judge_model,

        "judge_model_id":
            MODELS.get(
                judge_model,
                judge_model,
            ),

        "architecture_version":
            ARCHITECTURE_VERSION,

        "attempts":
            successful_attempt
            or JUDGE_MAX_ATTEMPTS,

        "evaluation_latency":
            total_latency,

        "content_origin":
            "creator_draft",

        "structured_requirements":
            json.dumps(
                normalized_requirements,
                ensure_ascii=False,
            ),

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
                gold_missing
            ),

        "negative_control":
            bool(
                gold.get(
                    "negative_control",
                    False,
                )
            ),
    }

    if evaluation is None:
        return (
            {
                "evaluation":
                    None,

                "blocking_flag":
                    None,

                "requirement_flag":
                    None,

                "requirement_signature":
                    None,

                "requirement_match":
                    {
                        "detected_gold_count":
                            0,

                        "matched_predicted_count":
                            0,

                        "extra_predicted_count":
                            0,
                    },

                "creator_regression":
                    None,
            },
            {
                **base,
                "status":
                    "FAILED",

                "error":
                    error
                    or "",
            },
        )

    compliance = (
        evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    requirements_found = (
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    advisories = (
        evaluation.get(
            "advisory_findings",
            [],
        )
        or []
    )

    (
        detected_block_count,
        detected_block_ids,
    ) = count_gold_findings_detected(
        gold_blocking,
        compliance,
    )

    req_match = (
        match_requirements(
            gold_missing,
            requirements_found,
        )
    )

    (
        basis_counts,
        relabels,
        copy_like,
    ) = advisory_diagnostics(
        evaluation
    )

    block_count = (
        blocking_count(
            evaluation
        )
    )

    req_count = (
        requirement_count(
            evaluation
        )
    )

    req_signature = (
        requirement_signature(
            evaluation
        )
    )

    authenticity_regression = (
        creator_regression(
            evaluation
        )
    )

    result = {
        "evaluation":
            evaluation,

        "blocking_flag":
            block_count > 0,

        "requirement_flag":
            req_count > 0,

        "requirement_signature":
            req_signature,

        "requirement_match":
            req_match,

        "creator_regression":
            authenticity_regression,
    }

    row = {
        **base,

        "status":
            "SUCCESS",

        "predicted_blocking_flag":
            block_count > 0,

        "predicted_blocking_count":
            block_count,

        "detected_gold_finding_count":
            detected_block_count,

        "detected_gold_issue_ids":
            json.dumps(
                detected_block_ids,
                ensure_ascii=False,
            ),

        "predicted_requirement_flag":
            req_count > 0,

        "predicted_requirement_count":
            req_count,

        "detected_gold_requirement_count":
            req_match[
                "detected_gold_count"
            ],

        "matched_predicted_requirement_count":
            req_match[
                "matched_predicted_count"
            ],

        "extra_predicted_requirement_count":
            req_match[
                "extra_predicted_count"
            ],

        "requirement_signature":
            json.dumps(
                list(
                    req_signature
                ),
                ensure_ascii=False,
            ),

        "mandatory_action_status":
            evaluation.get(
                "mandatory_action_status",
                "",
            ),

        "creator_authenticity_regression":
            authenticity_regression,

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
            heuristic_score(
                evaluation
            ),

        "advisory_count":
            len(
                advisories
            ),

        "advisory_supplied_context_count":
            basis_counts[
                "SUPPLIED_CONTEXT"
            ],

        "advisory_general_heuristic_count":
            basis_counts[
                "GENERAL_HEURISTIC"
            ],

        "advisory_system_grounding_review_count":
            basis_counts[
                "SYSTEM_GROUNDING_REVIEW"
            ],

        "advisory_other_basis_count":
            basis_counts[
                "OTHER"
            ],

        "advisory_provenance_relabel_count":
            relabels,

        "advisory_copy_like_suggestion_count":
            copy_like,

        "compliance_findings":
            json.dumps(
                compliance,
                ensure_ascii=False,
            ),

        "requirement_findings":
            json.dumps(
                requirements_found,
                ensure_ascii=False,
            ),

        "advisory_findings":
            json.dumps(
                advisories,
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

    return (
        result,
        row,
    )


def aggregate_panel(
    results,
):
    valid = {
        model:
            result

        for model, result
        in results.items()

        if result.get(
            "evaluation"
        ) is not None
    }

    if len(
        valid
    ) != len(
        JUDGE_MODELS
    ):
        return {
            "panel_complete":
                False,

            "blocking_agreement":
                None,

            "requirement_agreement":
                None,

            "compliance_decision":
                "INCOMPLETE_PANEL",

            "mandatory_decision":
                "INCOMPLETE_PANEL",

            "predicted_action":
                "INCOMPLETE",

            "human_review":
                False,

            "avg_heuristic":
                None,

            "avg_risk":
                None,

            "score_gap":
                None,
        }

    block_flags = [
        bool(
            valid[model][
                "blocking_flag"
            ]
        )
        for model
        in JUDGE_MODELS
    ]

    req_sets = [
        tuple(
            valid[model][
                "requirement_signature"
            ]
            or ()
        )
        for model
        in JUDGE_MODELS
    ]

    block_agree = (
        len(
            set(
                block_flags
            )
        )
        == 1
    )

    req_agree = (
        req_sets[0]
        == req_sets[1]
    )

    if all(
        block_flags
    ):
        compliance_decision = (
            "CONSENSUS_BLOCKING"
        )

    elif not any(
        block_flags
    ):
        compliance_decision = (
            "CONSENSUS_NO_BLOCKING"
        )

    else:
        compliance_decision = (
            "JUDGE_DISAGREEMENT"
        )

    if not block_agree:
        (
            mandatory_decision,
            predicted_action,
            human_review,
        ) = (
            "BLOCKING_DISAGREEMENT",
            "HUMAN_REVIEW",
            True,
        )

    elif not req_agree:
        (
            mandatory_decision,
            predicted_action,
            human_review,
        ) = (
            "REQUIREMENT_DISAGREEMENT",
            "HUMAN_REVIEW",
            True,
        )

    else:
        has_block = (
            block_flags[0]
        )

        has_req = bool(
            req_sets[0]
        )

        human_review = (
            False
        )

        if (
            has_block
            and has_req
        ):
            mandatory_decision = (
                "CONSENSUS_COMPLIANCE_AND_REQUIREMENT"
            )

            predicted_action = (
                "COMPLIANCE_AND_REQUIREMENT_ACTION"
            )

        elif has_block:
            mandatory_decision = (
                "CONSENSUS_COMPLIANCE"
            )

            predicted_action = (
                "COMPLIANCE_ACTION"
            )

        elif has_req:
            mandatory_decision = (
                "CONSENSUS_REQUIREMENT"
            )

            predicted_action = (
                "REQUIREMENT_ACTION"
            )

        else:
            mandatory_decision = (
                "CONSENSUS_NO_MANDATORY_ACTION"
            )

            predicted_action = (
                "NO_MANDATORY_ACTION"
            )

    scores = [
        heuristic_score(
            valid[model][
                "evaluation"
            ]
        )
        for model
        in JUDGE_MODELS
    ]

    risks = [
        valid[model][
            "evaluation"
        ].get(
            "unsupported_claim_risk"
        )
        for model
        in JUDGE_MODELS
    ]

    return {
        "panel_complete":
            True,

        "blocking_agreement":
            block_agree,

        "requirement_agreement":
            req_agree,

        "compliance_decision":
            compliance_decision,

        "mandatory_decision":
            mandatory_decision,

        "predicted_action":
            predicted_action,

        "human_review":
            human_review,

        "avg_heuristic":
            mean(
                scores
            ),

        "avg_risk":
            mean(
                risks
            ),

        "score_gap":
            round(
                abs(
                    float(
                        scores[0]
                    )
                    - float(
                        scores[1]
                    )
                ),
                2,
            ),
    }


def run_case(
    case,
    case_index,
    judge_rows,
):
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

    gold = (
        case.get(
            "gold",
            {},
        )
        or {}
    )

    gold_missing = (
        gold.get(
            "required_missing",
            [],
        )
        or []
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"[{case_index}] "
        f"{case_id} | "
        f"{case.get('scenario_name', '')}"
    )

    print(
        "Scenario type: "
        f"{case.get('scenario_type', '')}"
    )

    print(
        "=" * 80
    )

    results = {}

    for judge in JUDGE_MODELS:
        (
            result,
            row,
        ) = run_judge(
            case,
            case_index,
            judge,
        )

        results[
            judge
        ] = result

        judge_rows.append(
            row
        )

        save_csv(
            judge_rows,
            JUDGE_RESULTS_FILE,
        )

    panel = (
        aggregate_panel(
            results
        )
    )

    expected = (
        expected_action(
            gold
        )
    )

    req_exact = (
        False
    )

    if panel[
        "panel_complete"
    ]:
        step_match = (
            results[
                "step"
            ][
                "requirement_match"
            ]
        )

        qwen_match = (
            results[
                "qwen"
            ][
                "requirement_match"
            ]
        )

        gold_count = len(
            gold_missing
        )

        req_exact = bool(
            panel[
                "requirement_agreement"
            ]
            and step_match[
                "detected_gold_count"
            ]
            == gold_count
            and qwen_match[
                "detected_gold_count"
            ]
            == gold_count
            and step_match[
                "extra_predicted_count"
            ]
            == 0
            and qwen_match[
                "extra_predicted_count"
            ]
            == 0
        )

    return {
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

        "architecture_version":
            ARCHITECTURE_VERSION,

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
                gold_missing
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

        "gold_desired_product_action":
            gold.get(
                "desired_product_action",
                "",
            ),

        "expected_mandatory_action":
            expected,

        "panel_complete":
            panel[
                "panel_complete"
            ],

        "blocking_agreement":
            panel[
                "blocking_agreement"
            ],

        "requirement_agreement":
            panel[
                "requirement_agreement"
            ],

        "compliance_decision":
            panel[
                "compliance_decision"
            ],

        "mandatory_decision":
            panel[
                "mandatory_decision"
            ],

        "predicted_mandatory_action":
            panel[
                "predicted_action"
            ],

        "mandatory_route_correct":
            bool(
                panel[
                    "panel_complete"
                ]
                and panel[
                    "predicted_action"
                ]
                == expected
            ),

        "human_review_recommended":
            panel[
                "human_review"
            ],

        "consensus_requirement_exact_match":
            req_exact,

        "avg_heuristic_score":
            panel[
                "avg_heuristic"
            ],

        "avg_claim_risk":
            panel[
                "avg_risk"
            ],

        "judge_score_gap":
            panel[
                "score_gap"
            ],

        "step_blocking_flag":
            results[
                "step"
            ].get(
                "blocking_flag"
            ),

        "qwen_blocking_flag":
            results[
                "qwen"
            ].get(
                "blocking_flag"
            ),

        "step_requirement_flag":
            results[
                "step"
            ].get(
                "requirement_flag"
            ),

        "qwen_requirement_flag":
            results[
                "qwen"
            ].get(
                "requirement_flag"
            ),

        "step_creator_authenticity_regression":
            results[
                "step"
            ].get(
                "creator_regression"
            ),

        "qwen_creator_authenticity_regression":
            results[
                "qwen"
            ].get(
                "creator_regression"
            ),

        "creator_draft":
            case.get(
                "creator_draft",
                "",
            ),
    }


def calculate_metrics(
    cases,
    case_rows,
    judge_rows,
):
    cases_df = pd.DataFrame(
        case_rows
    )

    judges_df = pd.DataFrame(
        judge_rows
    )

    success = judges_df[
        judges_df[
            "status"
        ]
        == "SUCCESS"
    ].copy()

    complete = cases_df[
        cases_df[
            "panel_complete"
        ]
        == True
    ].copy()

    block_pos = complete[
        complete[
            "gold_blocking_expected"
        ]
        == True
    ]

    block_neg = complete[
        complete[
            "gold_blocking_expected"
        ]
        == False
    ]

    neg_controls = complete[
        complete[
            "negative_control"
        ]
        == True
    ]

    advisory_only = complete[
        complete[
            "scenario_type"
        ]
        == "advisory_only"
    ]

    individual_block_pos = success[
        success[
            "gold_blocking_expected"
        ]
        == True
    ]

    individual_block_neg = success[
        success[
            "gold_blocking_expected"
        ]
        == False
    ]

    req_pos = complete[
        complete[
            "gold_required_missing_count"
        ]
        > 0
    ]

    req_neg = complete[
        complete[
            "gold_required_missing_count"
        ]
        == 0
    ]

    individual_req_pos = success[
        success[
            "gold_required_missing_count"
        ]
        > 0
    ]

    individual_req_neg = success[
        success[
            "gold_required_missing_count"
        ]
        == 0
    ]

    total_gold_block_findings = int(
        individual_block_pos[
            "gold_blocking_issue_count"
        ]
        .fillna(0)
        .sum()
    )

    total_detected_block_findings = int(
        individual_block_pos[
            "detected_gold_finding_count"
        ]
        .fillna(0)
        .sum()
    )

    total_gold_req = int(
        success[
            "gold_required_missing_count"
        ]
        .fillna(0)
        .sum()
    )

    total_detected_req = int(
        success[
            "detected_gold_requirement_count"
        ]
        .fillna(0)
        .sum()
    )

    total_predicted_req = int(
        success[
            "predicted_requirement_count"
        ]
        .fillna(0)
        .sum()
    )

    total_matched_req = int(
        success[
            "matched_predicted_requirement_count"
        ]
        .fillna(0)
        .sum()
    )

    creator_calls = success[
        success[
            "case_id"
        ].isin(
            CREATOR_SEMANTICS_CASE_IDS
        )
    ]

    creator_regressions = int(
        creator_calls[
            "creator_authenticity_regression"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    total_advisories = int(
        success[
            "advisory_count"
        ]
        .fillna(0)
        .sum()
    )

    supplied = int(
        success[
            "advisory_supplied_context_count"
        ]
        .fillna(0)
        .sum()
    )

    heuristic = int(
        success[
            "advisory_general_heuristic_count"
        ]
        .fillna(0)
        .sum()
    )

    grounding = int(
        success[
            "advisory_system_grounding_review_count"
        ]
        .fillna(0)
        .sum()
    )

    other = int(
        success[
            "advisory_other_basis_count"
        ]
        .fillna(0)
        .sum()
    )

    relabels = int(
        success[
            "advisory_provenance_relabel_count"
        ]
        .fillna(0)
        .sum()
    )

    copy_like = int(
        success[
            "advisory_copy_like_suggestion_count"
        ]
        .fillna(0)
        .sum()
    )

    per_judge = {}

    for judge in JUDGE_MODELS:
        data = success[
            success[
                "judge_model"
            ]
            == judge
        ]

        bp = data[
            data[
                "gold_blocking_expected"
            ]
            == True
        ]

        bn = data[
            data[
                "gold_blocking_expected"
            ]
            == False
        ]

        rp = data[
            data[
                "gold_required_missing_count"
            ]
            > 0
        ]

        rn = data[
            data[
                "gold_required_missing_count"
            ]
            == 0
        ]

        creator_data = data[
            data[
                "case_id"
            ].isin(
                CREATOR_SEMANTICS_CASE_IDS
            )
        ]

        per_judge[
            judge
        ] = {
            "successful_calls":
                int(
                    len(data)
                ),

            "blocking_case_recall":
                safe_rate(
                    int(
                        bp[
                            "predicted_blocking_flag"
                        ]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                    len(bp),
                ),

            "false_blocking_rate":
                safe_rate(
                    int(
                        bn[
                            "predicted_blocking_flag"
                        ]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                    len(bn),
                ),

            "blocking_finding_recall":
                safe_rate(
                    int(
                        bp[
                            "detected_gold_finding_count"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                    int(
                        bp[
                            "gold_blocking_issue_count"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                ),

            "requirement_case_recall":
                safe_rate(
                    int(
                        (
                            rp[
                                "detected_gold_requirement_count"
                            ]
                            > 0
                        ).sum()
                    ),
                    len(rp),
                ),

            "false_requirement_case_rate":
                safe_rate(
                    int(
                        rn[
                            "predicted_requirement_flag"
                        ]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                    len(rn),
                ),

            "requirement_finding_recall":
                safe_rate(
                    int(
                        data[
                            "detected_gold_requirement_count"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                    int(
                        data[
                            "gold_required_missing_count"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                ),

            "requirement_finding_precision":
                safe_rate(
                    int(
                        data[
                            "matched_predicted_requirement_count"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                    int(
                        data[
                            "predicted_requirement_count"
                        ]
                        .fillna(0)
                        .sum()
                    ),
                ),

            "creator_semantics_clear_rate":
                safe_rate(
                    (
                        len(
                            creator_data
                        )
                        - int(
                            creator_data[
                                "creator_authenticity_regression"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        )
                    ),
                    len(
                        creator_data
                    ),
                ),

            "median_evaluation_latency_seconds":
                median(
                    data[
                        "evaluation_latency"
                    ].tolist()
                ),

            "avg_evaluation_latency_seconds":
                mean(
                    data[
                        "evaluation_latency"
                    ].tolist()
                ),

            "retry_call_count":
                int(
                    (
                        data[
                            "attempts"
                        ]
                        .fillna(1)
                        .astype(int)
                        > 1
                    ).sum()
                ),
        }

    metrics = {
        "benchmark":
            "Creator Draft Compliance Stress Suite",

        "evaluation_architecture":
            ARCHITECTURE_VERSION,

        "generator_used":
            False,

        "reviser_used":
            False,

        "numerical_quality_threshold_used":
            False,

        "total_cases":
            len(
                cases
            ),

        "completed_case_panels":
            len(
                complete
            ),

        "total_expected_judge_calls":
            len(
                cases
            )
            * len(
                JUDGE_MODELS
            ),

        "successful_judge_calls":
            len(
                success
            ),

        "compliance":
            {
                "consensus_blocking_recall_for_auto_fix":
                    safe_rate(
                        len(
                            block_pos[
                                block_pos[
                                    "compliance_decision"
                                ]
                                == "CONSENSUS_BLOCKING"
                            ]
                        ),
                        len(
                            block_pos
                        ),
                    ),

                "false_consensus_blocking_rate":
                    safe_rate(
                        len(
                            block_neg[
                                block_neg[
                                    "compliance_decision"
                                ]
                                == "CONSENSUS_BLOCKING"
                            ]
                        ),
                        len(
                            block_neg
                        ),
                    ),

                "cross_judge_blocking_agreement_rate":
                    safe_rate(
                        int(
                            complete[
                                "blocking_agreement"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            complete
                        ),
                    ),

                "human_blocking_disagreement_rate":
                    safe_rate(
                        len(
                            complete[
                                complete[
                                    "compliance_decision"
                                ]
                                == "JUDGE_DISAGREEMENT"
                            ]
                        ),
                        len(
                            complete
                        ),
                    ),

                "negative_control_clear_rate":
                    safe_rate(
                        len(
                            neg_controls[
                                neg_controls[
                                    "compliance_decision"
                                ]
                                == "CONSENSUS_NO_BLOCKING"
                            ]
                        ),
                        len(
                            neg_controls
                        ),
                    ),

                "advisory_only_clear_rate":
                    safe_rate(
                        len(
                            advisory_only[
                                advisory_only[
                                    "compliance_decision"
                                ]
                                == "CONSENSUS_NO_BLOCKING"
                            ]
                        ),
                        len(
                            advisory_only
                        ),
                    ),

                "individual_judge_blocking_case_recall":
                    safe_rate(
                        int(
                            individual_block_pos[
                                "predicted_blocking_flag"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            individual_block_pos
                        ),
                    ),

                "individual_judge_false_blocking_rate":
                    safe_rate(
                        int(
                            individual_block_neg[
                                "predicted_blocking_flag"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            individual_block_neg
                        ),
                    ),

                "individual_judge_blocking_finding_recall":
                    safe_rate(
                        total_detected_block_findings,
                        total_gold_block_findings,
                    ),
            },

        "requirements":
            {
                "structured_requirement_findings_supported":
                    True,

                "consensus_requirement_exact_case_recall":
                    safe_rate(
                        int(
                            req_pos[
                                "consensus_requirement_exact_match"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            req_pos
                        ),
                    ),

                "false_consensus_requirement_action_rate":
                    safe_rate(
                        len(
                            req_neg[
                                req_neg[
                                    "predicted_mandatory_action"
                                ].isin(
                                    [
                                        "REQUIREMENT_ACTION",
                                        "COMPLIANCE_AND_REQUIREMENT_ACTION",
                                    ]
                                )
                            ]
                        ),
                        len(
                            req_neg
                        ),
                    ),

                "cross_judge_requirement_agreement_rate":
                    safe_rate(
                        int(
                            complete[
                                "requirement_agreement"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            complete
                        ),
                    ),

                "individual_judge_requirement_case_recall":
                    safe_rate(
                        int(
                            (
                                individual_req_pos[
                                    "detected_gold_requirement_count"
                                ]
                                > 0
                            ).sum()
                        ),
                        len(
                            individual_req_pos
                        ),
                    ),

                "individual_judge_false_requirement_case_rate":
                    safe_rate(
                        int(
                            individual_req_neg[
                                "predicted_requirement_flag"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            individual_req_neg
                        ),
                    ),

                "individual_judge_requirement_finding_recall":
                    safe_rate(
                        total_detected_req,
                        total_gold_req,
                    ),

                "individual_judge_requirement_finding_precision":
                    safe_rate(
                        total_matched_req,
                        total_predicted_req,
                    ),
            },

        "mandatory_routing":
            {
                "routing_accuracy_on_complete_panels":
                    safe_rate(
                        int(
                            complete[
                                "mandatory_route_correct"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            complete
                        ),
                    ),

                "human_review_escalation_rate":
                    safe_rate(
                        int(
                            complete[
                                "human_review_recommended"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        ),
                        len(
                            complete
                        ),
                    ),
            },

        "creator_semantics":
            {
                "controlled_case_ids":
                    sorted(
                        CREATOR_SEMANTICS_CASE_IDS
                    ),

                "authenticity_regression_calls":
                    creator_regressions,

                "creator_semantics_clear_rate":
                    safe_rate(
                        (
                            len(
                                creator_calls
                            )
                            - creator_regressions
                        ),
                        len(
                            creator_calls
                        ),
                    ),

                "note":
                    (
                        "Targeted regression diagnostic, "
                        "not general authenticity accuracy."
                    ),
            },

        "advisory_provenance":
            {
                "total_advisory_findings":
                    total_advisories,

                "supplied_context_count":
                    supplied,

                "general_heuristic_count":
                    heuristic,

                "system_grounding_review_count":
                    grounding,

                "other_basis_count":
                    other,

                "provenance_relabel_count":
                    relabels,

                "valid_known_basis_rate":
                    safe_rate(
                        (
                            supplied
                            + heuristic
                            + grounding
                        ),
                        total_advisories,
                    ),

                "advisory_copy_like_suggestion_count":
                    copy_like,

                "advisory_copy_like_suggestion_rate":
                    safe_rate(
                        copy_like,
                        total_advisories,
                    ),

                "note":
                    (
                        "Copy-like suggestion detection "
                        "is heuristic diagnostic only."
                    ),
            },

        "per_judge":
            per_judge,
    }

    metrics[
        "baseline_comparison"
    ] = baseline_comparison(
        metrics
    )

    return metrics


def baseline_comparison(
    metrics,
):
    if not BASELINE_METRICS_FILE.exists():
        return {
            "available":
                False,

            "reason":
                "Baseline metrics file not found.",
        }

    try:
        baseline = json.loads(
            BASELINE_METRICS_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        return {
            "available":
                False,

            "reason":
                str(
                    exc
                ),
        }

    compliance = (
        metrics[
            "compliance"
        ]
    )

    requirements = (
        metrics[
            "requirements"
        ]
    )

    return {
        "available":
            True,

        "consensus_blocking_recall_for_auto_fix":
            {
                "baseline":
                    baseline.get(
                        "consensus_blocking_recall_for_auto_fix"
                    ),

                "v2":
                    compliance[
                        "consensus_blocking_recall_for_auto_fix"
                    ],
            },

        "false_consensus_blocking_rate":
            {
                "baseline":
                    baseline.get(
                        "false_consensus_blocking_rate"
                    ),

                "v2":
                    compliance[
                        "false_consensus_blocking_rate"
                    ],
            },

        "cross_judge_blocking_agreement_rate":
            {
                "baseline":
                    baseline.get(
                        "cross_judge_blocking_agreement_rate"
                    ),

                "v2":
                    compliance[
                        "cross_judge_blocking_agreement_rate"
                    ],
            },

        "individual_judge_blocking_finding_recall":
            {
                "baseline":
                    baseline.get(
                        "individual_judge_blocking_finding_recall"
                    ),

                "v2":
                    compliance[
                        "individual_judge_blocking_finding_recall"
                    ],
            },

        "formal_requirement_finding_recall":
            {
                "baseline":
                    None,

                "v2":
                    requirements[
                        "individual_judge_requirement_finding_recall"
                    ],

                "baseline_interpretation":
                    (
                        "N/A: baseline had no structured "
                        "Requirement layer."
                    ),
            },
    }


def print_summary(
    metrics,
):
    compliance = (
        metrics[
            "compliance"
        ]
    )

    requirements = (
        metrics[
            "requirements"
        ]
    )

    routing = (
        metrics[
            "mandatory_routing"
        ]
    )

    creator = (
        metrics[
            "creator_semantics"
        ]
    )

    advisory = (
        metrics[
            "advisory_provenance"
        ]
    )

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "CREATOR DRAFT STRESS v2.2 SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "Completed panels: "
        f"{metrics['completed_case_panels']}/"
        f"{metrics['total_cases']}"
    )

    print(
        "Successful Judge calls: "
        f"{metrics['successful_judge_calls']}/"
        f"{metrics['total_expected_judge_calls']}"
    )

    print(
        "\nCompliance"
    )

    print(
        "  Consensus Blocking Recall:",
        format_rate(
            compliance[
                "consensus_blocking_recall_for_auto_fix"
            ]
        ),
    )

    print(
        "  False Consensus Blocking Rate:",
        format_rate(
            compliance[
                "false_consensus_blocking_rate"
            ]
        ),
    )

    print(
        "  Cross-Judge Blocking Agreement:",
        format_rate(
            compliance[
                "cross_judge_blocking_agreement_rate"
            ]
        ),
    )

    print(
        "  Blocking Finding Recall:",
        format_rate(
            compliance[
                "individual_judge_blocking_finding_recall"
            ]
        ),
    )

    print(
        "\nRequirements"
    )

    print(
        "  Consensus Exact Case Recall:",
        format_rate(
            requirements[
                "consensus_requirement_exact_case_recall"
            ]
        ),
    )

    print(
        "  False Consensus Requirement Action:",
        format_rate(
            requirements[
                "false_consensus_requirement_action_rate"
            ]
        ),
    )

    print(
        "  Cross-Judge Requirement Agreement:",
        format_rate(
            requirements[
                "cross_judge_requirement_agreement_rate"
            ]
        ),
    )

    print(
        "  Requirement Finding Recall:",
        format_rate(
            requirements[
                "individual_judge_requirement_finding_recall"
            ]
        ),
    )

    print(
        "  Requirement Finding Precision:",
        format_rate(
            requirements[
                "individual_judge_requirement_finding_precision"
            ]
        ),
    )

    print(
        "\nMandatory Routing"
    )

    print(
        "  Routing Accuracy:",
        format_rate(
            routing[
                "routing_accuracy_on_complete_panels"
            ]
        ),
    )

    print(
        "  Human Review Escalation:",
        format_rate(
            routing[
                "human_review_escalation_rate"
            ]
        ),
    )

    print(
        "\nCreator Semantics"
    )

    print(
        "  Clear Rate:",
        format_rate(
            creator[
                "creator_semantics_clear_rate"
            ]
        ),
    )

    print(
        "  Authenticity Regression Calls:",
        creator[
            "authenticity_regression_calls"
        ],
    )

    print(
        "\nAdvisory Provenance"
    )

    print(
        "  Total:",
        advisory[
            "total_advisory_findings"
        ],
    )

    print(
        "  SUPPLIED_CONTEXT:",
        advisory[
            "supplied_context_count"
        ],
    )

    print(
        "  GENERAL_HEURISTIC:",
        advisory[
            "general_heuristic_count"
        ],
    )

    print(
        "  SYSTEM_GROUNDING_REVIEW:",
        advisory[
            "system_grounding_review_count"
        ],
    )

    print(
        "  Provenance Relabels:",
        advisory[
            "provenance_relabel_count"
        ],
    )

    print(
        "  Copy-like Suggestions:",
        advisory[
            "advisory_copy_like_suggestion_count"
        ],
    )

    print(
        "\nPer Judge"
    )

    for (
        judge,
        values,
    ) in metrics[
        "per_judge"
    ].items():
        print(
            f"  {judge}: "
            f"blocking="
            f"{format_rate(values['blocking_case_recall'])}, "
            f"finding="
            f"{format_rate(values['blocking_finding_recall'])}, "
            f"req_recall="
            f"{format_rate(values['requirement_finding_recall'])}, "
            f"req_precision="
            f"{format_rate(values['requirement_finding_precision'])}, "
            f"creator="
            f"{format_rate(values['creator_semantics_clear_rate'])}, "
            f"median_latency="
            f"{values['median_evaluation_latency_seconds']}s, "
            f"retries="
            f"{values['retry_call_count']}"
        )

    comparison = (
        metrics.get(
            "baseline_comparison",
            {},
        )
    )

    if comparison.get(
        "available"
    ):
        print(
            "\nBaseline -> v2.2"
        )

        for (
            key,
            item,
        ) in comparison.items():

            if key == "available":
                continue

            print(
                f"  {key}: "
                f"{format_rate(item.get('baseline'))} "
                f"-> "
                f"{format_rate(item.get('v2'))}"
            )

    print(
        "\nResults"
    )

    print(
        " ",
        CASE_RESULTS_FILE,
    )

    print(
        " ",
        JUDGE_RESULTS_FILE,
    )

    print(
        " ",
        METRICS_FILE,
    )


if __name__ == "__main__":
    cases = load_cases()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GrowthPilot Creator Draft Stress Benchmark v2.2"
    )

    print(
        "=" * 80
    )

    print(
        f"Cases: {len(cases)} "
        f"| Judges: {JUDGE_MODELS} "
        f"| Max attempts: {JUDGE_MAX_ATTEMPTS}"
    )

    print(
        "Generator: NOT USED "
        "| Reviser: NOT USED "
        "| Threshold: None"
    )

    print(
        "Reuses frozen baseline context/matching helpers "
        "for Before/After fairness."
    )

    case_rows = []
    judge_rows = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        try:
            row = run_case(
                case,
                index,
                judge_rows,
            )

        except Exception as exc:
            case_id = (
                case.get(
                    "case_id"
                )
                or f"STRESS_{index:03d}"
            )

            print(
                f"\n❌ Unexpected failure for "
                f"{case_id}: {exc}"
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

                "mandatory_decision":
                    "FAILED_UNEXPECTED",

                "predicted_mandatory_action":
                    "INCOMPLETE",

                "mandatory_route_correct":
                    False,

                "human_review_recommended":
                    False,

                "error":
                    str(
                        exc
                    ),
            }

        case_rows.append(
            row
        )

        save_csv(
            case_rows,
            CASE_RESULTS_FILE,
        )

    metrics = calculate_metrics(
        cases,
        case_rows,
        judge_rows,
    )

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(
        metrics
    )