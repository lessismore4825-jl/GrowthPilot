import json
import time
from pathlib import Path

import pandas as pd

from src.generator import generate_content
from src.evaluator import evaluate_content
from src.reviser import revise_content
from src.llm_client import MODELS


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation_cases.json"
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

BATCH_RESULTS_FILE = (
    RESULTS_DIR
    / "batch_results.csv"
)

JUDGE_RESULTS_FILE = (
    RESULTS_DIR
    / "judge_results.csv"
)


# =========================================================
# Experiment Configuration
# =========================================================

CANDIDATE_MODELS = [
    "step",
    "qwen",
]

JUDGE_MODELS = [
    "step",
    "qwen",
]


QUALITY_THRESHOLD = 9.0

CLAIM_RISK_THRESHOLD = 2


# Judge can try:
#
# Attempt 1
# ↓ failure
# Attempt 2
#
# If both fail, that Judge is marked FAILED.
#
JUDGE_MAX_ATTEMPTS = 2


# =========================================================
# Evaluation Metrics
# =========================================================

SCORE_FIELDS = [
    "brand_alignment",
    "tone_match",
    "selling_point_coverage",
    "factual_consistency",
    "unsupported_claim_risk",
    "overall_score",
]


# =========================================================
# Load Evaluation Cases
# =========================================================

def load_cases() -> list:
    """
    Load evaluation cases from data/evaluation_cases.json.

    Supports either:

    [
        {...},
        {...}
    ]

    or:

    {
        "cases": [
            {...},
            {...}
        ]
    }
    """

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{DATA_FILE}"
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
            "evaluation_cases.json must contain "
            "a list of cases or a "
            "{'cases': [...]} object."
        )


    if not cases:

        raise ValueError(
            "No evaluation cases were found."
        )


    return cases


# =========================================================
# Case Helpers
# =========================================================

def get_case_id(
    case: dict,
    index: int,
) -> str:
    """
    Return the case ID.
    """

    return (
        case.get("case_id")
        or case.get("id")
        or f"CASE_{index:03d}"
    )


def get_challenge(
    case: dict,
) -> str:
    """
    Return the case challenge / description
    when available.
    """

    return (
        case.get("challenge")
        or case.get("description")
        or ""
    )


def validate_case(
    case: dict,
    case_id: str,
):
    """
    Validate required case fields.
    """

    required_fields = [
        "brand_info",
        "campaign_brief",
    ]


    for field in required_fields:

        if field not in case:

            raise ValueError(
                f"{case_id} is missing "
                f"required field: {field}"
            )


# =========================================================
# Generic Timing Helper
# =========================================================

def timed_call(
    function,
    **kwargs,
):
    """
    Execute a function and return:

    result,
    latency_seconds
    """

    start_time = time.perf_counter()

    result = function(
        **kwargs
    )

    latency = (
        time.perf_counter()
        - start_time
    )


    return (
        result,
        round(
            latency,
            2,
        ),
    )


# =========================================================
# Cross-Judge Evaluation
# =========================================================

def run_all_judges(
    case_id: str,
    challenge: str,
    candidate_model: str,
    version: str,
    brand_info: str,
    campaign_brief: str,
    content: str,
    judge_rows: list,
) -> dict:
    """
    Evaluate one piece of content with every Judge.

    Current design:

    Candidate Output
        ↓
    ├── Step Judge
    └── Qwen Judge

    Each Judge can attempt evaluation up to
    JUDGE_MAX_ATTEMPTS times.

    Returns:

    {
        "step": {
            "evaluation": {...},
            "latency": 12.3,
            "error": None,
            "attempts": 1
        },

        "qwen": {
            ...
        }
    }
    """

    results = {}


    for judge_model in JUDGE_MODELS:

        print(
            f"\n      Judge: "
            f"{judge_model}"
        )


        evaluation = None

        final_error = None

        successful_attempt = None

        total_judge_latency = 0.0


        # =================================================
        # Retry Loop
        # =================================================

        for attempt in range(
            1,
            JUDGE_MAX_ATTEMPTS + 1,
        ):

            print(
                f"      Attempt "
                f"{attempt}/"
                f"{JUDGE_MAX_ATTEMPTS}"
            )


            attempt_start = (
                time.perf_counter()
            )


            try:

                evaluation = evaluate_content(
                    brand_info=brand_info,

                    campaign_brief=campaign_brief,

                    generated_content=content,

                    judge_model_key=judge_model,
                )


                attempt_latency = (
                    time.perf_counter()
                    - attempt_start
                )


                total_judge_latency += (
                    attempt_latency
                )


                successful_attempt = attempt

                final_error = None


                print(
                    f"      ✅ "
                    f"Overall="
                    f"{evaluation['overall_score']}"
                    f" | Risk="
                    f"{evaluation['unsupported_claim_risk']}"
                    f" | Attempt latency="
                    f"{attempt_latency:.2f}s"
                )


                break


            except Exception as error:

                attempt_latency = (
                    time.perf_counter()
                    - attempt_start
                )


                total_judge_latency += (
                    attempt_latency
                )


                final_error = str(
                    error
                )


                print(
                    f"      ⚠️ Attempt "
                    f"{attempt} failed "
                    f"after "
                    f"{attempt_latency:.2f}s: "
                    f"{error}"
                )


                if (
                    attempt
                    < JUDGE_MAX_ATTEMPTS
                ):

                    print(
                        "      Retrying Judge..."
                    )


        total_judge_latency = round(
            total_judge_latency,
            2,
        )


        # =================================================
        # Judge SUCCESS
        # =================================================

        if evaluation is not None:

            results[
                judge_model
            ] = {
                "evaluation":
                    evaluation,

                "latency":
                    total_judge_latency,

                "error":
                    None,

                "attempts":
                    successful_attempt,
            }


            judge_rows.append(
                {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "version":
                        version,

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
                        total_judge_latency,

                    "brand_alignment":
                        evaluation[
                            "brand_alignment"
                        ],

                    "tone_match":
                        evaluation[
                            "tone_match"
                        ],

                    "selling_point_coverage":
                        evaluation[
                            "selling_point_coverage"
                        ],

                    "factual_consistency":
                        evaluation[
                            "factual_consistency"
                        ],

                    "unsupported_claim_risk":
                        evaluation[
                            "unsupported_claim_risk"
                        ],

                    "overall_score":
                        evaluation[
                            "overall_score"
                        ],

                    "issues":
                        json.dumps(
                            evaluation.get(
                                "issues",
                                [],
                            ),
                            ensure_ascii=False,
                        ),

                    "suggestions":
                        json.dumps(
                            evaluation.get(
                                "suggestions",
                                [],
                            ),
                            ensure_ascii=False,
                        ),

                    "error":
                        "",
                }
            )


        # =================================================
        # Judge FAILED
        # =================================================

        else:

            print(
                f"      ❌ Judge failed "
                f"after "
                f"{JUDGE_MAX_ATTEMPTS} attempts."
            )


            results[
                judge_model
            ] = {
                "evaluation":
                    None,

                "latency":
                    total_judge_latency,

                "error":
                    final_error,

                "attempts":
                    JUDGE_MAX_ATTEMPTS,
            }


            judge_rows.append(
                {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "version":
                        version,

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
                        total_judge_latency,

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

                    "overall_score":
                        None,

                    "issues":
                        "",

                    "suggestions":
                        "",

                    "error":
                        final_error or "",
                }
            )


    return results


# =========================================================
# Cross-Judge Aggregation
# =========================================================

def aggregate_judges(
    judge_results: dict,
) -> dict:
    """
    Calculate Cross-Judge metrics.

    IMPORTANT:

    A valid Cross-Judge result requires
    ALL configured Judges to succeed.

    This prevents invalid comparisons such as:

    V1:
        only Qwen Judge

    V2:
        Step + Qwen Judges

    Those two averages would not be comparable.
    """

    valid_evaluations = [
        result["evaluation"]

        for result
        in judge_results.values()

        if result.get(
            "evaluation"
        ) is not None
    ]


    valid_judge_count = len(
        valid_evaluations
    )


    expected_judge_count = len(
        JUDGE_MODELS
    )


    panel_complete = (
        valid_judge_count
        == expected_judge_count
    )


    # =====================================================
    # Incomplete Judge Panel
    # =====================================================

    if not panel_complete:

        return {
            "valid_judge_count":
                valid_judge_count,

            "expected_judge_count":
                expected_judge_count,

            "panel_complete":
                False,

            "avg_overall_score":
                None,

            "avg_claim_risk":
                None,

            "judge_score_gap":
                None,
        }


    # =====================================================
    # Complete Judge Panel
    # =====================================================

    avg_overall_score = sum(
        evaluation[
            "overall_score"
        ]

        for evaluation
        in valid_evaluations
    ) / valid_judge_count


    avg_claim_risk = sum(
        evaluation[
            "unsupported_claim_risk"
        ]

        for evaluation
        in valid_evaluations
    ) / valid_judge_count


    # -----------------------------------------------------
    # Judge disagreement
    # -----------------------------------------------------

    step_score = (
        judge_results[
            "step"
        ][
            "evaluation"
        ][
            "overall_score"
        ]
    )


    qwen_score = (
        judge_results[
            "qwen"
        ][
            "evaluation"
        ][
            "overall_score"
        ]
    )


    judge_score_gap = abs(
        step_score
        - qwen_score
    )


    return {
        "valid_judge_count":
            valid_judge_count,

        "expected_judge_count":
            expected_judge_count,

        "panel_complete":
            True,

        "avg_overall_score":
            round(
                avg_overall_score,
                2,
            ),

        "avg_claim_risk":
            round(
                avg_claim_risk,
                2,
            ),

        "judge_score_gap":
            round(
                judge_score_gap,
                2,
            ),
    }


# =========================================================
# Benchmark Quality Gate
# =========================================================

def needs_revision(
    judge_results: dict,
) -> bool:
    """
    Decide whether V1 requires revision.

    Requires a complete Judge Panel.

    Revision is triggered when:

    1. Cross-Judge average overall score < 9.0

    OR

    2. Cross-Judge average unsupported claim risk > 2

    OR

    3. Either Judge identifies concrete issues.
    """

    aggregate = aggregate_judges(
        judge_results
    )


    if not aggregate[
        "panel_complete"
    ]:

        raise ValueError(
            "Cannot make revision decision "
            "because the Judge Panel is incomplete."
        )


    if (
        aggregate[
            "avg_overall_score"
        ] < QUALITY_THRESHOLD
    ):

        return True


    if (
        aggregate[
            "avg_claim_risk"
        ] > CLAIM_RISK_THRESHOLD
    ):

        return True


    for result in judge_results.values():

        evaluation = result.get(
            "evaluation"
        )


        if (
            evaluation
            and evaluation.get(
                "issues",
                [],
            )
        ):

            return True


    return False


# =========================================================
# Add Judge Scores to Main Row
# =========================================================

def add_judge_scores_to_row(
    row: dict,
    version: str,
    judge_results: dict,
):
    """
    Add each Judge's individual scores into
    the main wide-format CSV row.
    """

    for judge_model in JUDGE_MODELS:

        result = judge_results.get(
            judge_model,
            {},
        )


        evaluation = result.get(
            "evaluation"
        )


        latency = result.get(
            "latency"
        )


        error = result.get(
            "error"
        )


        attempts = result.get(
            "attempts"
        )


        prefix = (
            f"{version}_"
            f"{judge_model}_judge"
        )


        row[
            f"{prefix}_latency"
        ] = latency


        row[
            f"{prefix}_attempts"
        ] = attempts


        row[
            f"{prefix}_error"
        ] = error or ""


        if evaluation:

            for field in SCORE_FIELDS:

                row[
                    f"{prefix}_"
                    f"{field}"
                ] = evaluation[
                    field
                ]


            row[
                f"{prefix}_issues"
            ] = json.dumps(
                evaluation.get(
                    "issues",
                    [],
                ),
                ensure_ascii=False,
            )


        else:

            for field in SCORE_FIELDS:

                row[
                    f"{prefix}_"
                    f"{field}"
                ] = None


            row[
                f"{prefix}_issues"
            ] = ""


# =========================================================
# Save Results
# =========================================================

def save_results(
    batch_rows: list,
    judge_rows: list,
):
    """
    Save intermediate results after each
    Candidate × Case experiment.

    This prevents losing finished results if
    a later API request fails.
    """

    if batch_rows:

        pd.DataFrame(
            batch_rows
        ).to_csv(
            BATCH_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    if judge_rows:

        pd.DataFrame(
            judge_rows
        ).to_csv(
            JUDGE_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


# =========================================================
# Candidate × Case Experiment
# =========================================================

def run_candidate_case(
    case: dict,
    case_index: int,
    candidate_model: str,
    judge_rows: list,
) -> dict:
    """
    Run one complete experiment:

    Candidate generates V1
            ↓
    Step Judge + Qwen Judge
            ↓
    Cross-Judge Quality Gate
            ↓
    Candidate revises V2 if required
            ↓
    Step Judge + Qwen Judge
            ↓
    V1 vs V2 comparison
    """

    case_id = get_case_id(
        case,
        case_index,
    )


    challenge = get_challenge(
        case
    )


    validate_case(
        case,
        case_id,
    )


    brand_info = case[
        "brand_info"
    ]


    campaign_brief = case[
        "campaign_brief"
    ]


    row = {
        "case_id":
            case_id,

        "challenge":
            challenge,

        "candidate_model":
            candidate_model,

        "candidate_model_id":
            MODELS[
                candidate_model
            ],

        "status":
            "RUNNING",

        "error":
            "",
    }


    # =====================================================
    # Generate V1
    # =====================================================

    print(
        "\n"
        + "-" * 70
    )


    print(
        f"Generating V1 with "
        f"{candidate_model}..."
    )


    try:

        (
            v1_content,
            generation_latency,
        ) = timed_call(
            generate_content,

            brand_info=brand_info,

            campaign_brief=campaign_brief,

            model_key=candidate_model,
        )


        row[
            "v1_content"
        ] = v1_content


        row[
            "v1_generation_latency"
        ] = generation_latency


        print(
            f"✅ V1 generated "
            f"in "
            f"{generation_latency:.2f}s"
        )


    except Exception as error:

        print(
            f"❌ Generation failed: "
            f"{error}"
        )


        row[
            "status"
        ] = "FAILED_GENERATION"


        row[
            "error"
        ] = str(
            error
        )


        return row


    # =====================================================
    # Evaluate V1
    # =====================================================

    print(
        "\nEvaluating V1 "
        "with all Judges..."
    )


    v1_judges = run_all_judges(
        case_id=case_id,

        challenge=challenge,

        candidate_model=candidate_model,

        version="v1",

        brand_info=brand_info,

        campaign_brief=campaign_brief,

        content=v1_content,

        judge_rows=judge_rows,
    )


    add_judge_scores_to_row(
        row=row,

        version="v1",

        judge_results=v1_judges,
    )


    v1_aggregate = aggregate_judges(
        v1_judges
    )


    row[
        "v1_valid_judge_count"
    ] = v1_aggregate[
        "valid_judge_count"
    ]


    row[
        "v1_expected_judge_count"
    ] = v1_aggregate[
        "expected_judge_count"
    ]


    row[
        "v1_judge_panel_complete"
    ] = v1_aggregate[
        "panel_complete"
    ]


    row[
        "v1_cross_judge_overall"
    ] = v1_aggregate[
        "avg_overall_score"
    ]


    row[
        "v1_cross_judge_claim_risk"
    ] = v1_aggregate[
        "avg_claim_risk"
    ]


    row[
        "v1_judge_score_gap"
    ] = v1_aggregate[
        "judge_score_gap"
    ]


    # =====================================================
    # Stop if V1 Judge Panel is incomplete
    # =====================================================

    if not v1_aggregate[
        "panel_complete"
    ]:

        print(
            "\n❌ V1 Cross-Judge Panel "
            "is incomplete."
        )


        print(
            "Experiment cannot continue "
            "with a valid Cross-Judge score."
        )


        row[
            "revision_triggered"
        ] = None


        row[
            "status"
        ] = "FAILED_V1_JUDGES"


        row[
            "error"
        ] = (
            "One or more V1 Judges failed "
            "after all retry attempts."
        )


        return row


    # =====================================================
    # Revision Gate
    # =====================================================

    revision_required = needs_revision(
        v1_judges
    )


    row[
        "revision_triggered"
    ] = revision_required


    # =====================================================
    # No Revision Needed
    # =====================================================

    if not revision_required:

        print(
            "\n✅ V1 passed "
            "Cross-Judge Quality Gate."
        )


        row[
            "v2_content"
        ] = ""


        row[
            "v2_revision_latency"
        ] = None


        row[
            "v2_cross_judge_overall"
        ] = None


        row[
            "v2_cross_judge_claim_risk"
        ] = None


        row[
            "v2_judge_score_gap"
        ] = None


        row[
            "overall_improvement"
        ] = 0


        row[
            "claim_risk_reduction"
        ] = 0


        row[
            "revision_success"
        ] = None


        row[
            "status"
        ] = "SUCCESS_NO_REVISION"


        return row


    # =====================================================
    # Revision Triggered
    # =====================================================

    print(
        "\n⚠️ Revision triggered."
    )


    print(
        f"Revising with "
        f"{candidate_model}..."
    )


    # =====================================================
    # Combine Feedback From Both Judges
    # =====================================================

    combined_issues = []

    combined_suggestions = []


    for judge_model in JUDGE_MODELS:

        evaluation = (
            v1_judges
            .get(
                judge_model,
                {},
            )
            .get(
                "evaluation"
            )
        )


        if not evaluation:
            continue


        for issue in evaluation.get(
            "issues",
            [],
        ):

            if (
                issue
                not in combined_issues
            ):

                combined_issues.append(
                    issue
                )


        for suggestion in evaluation.get(
            "suggestions",
            [],
        ):

            if (
                suggestion
                not in combined_suggestions
            ):

                combined_suggestions.append(
                    suggestion
                )


    combined_feedback = {
        "issues":
            combined_issues,

        "suggestions":
            combined_suggestions,
    }


    # =====================================================
    # Generate V2
    # =====================================================

    try:

        (
            v2_content,
            revision_latency,
        ) = timed_call(
            revise_content,

            brand_info=brand_info,

            campaign_brief=campaign_brief,

            original_content=v1_content,

            evaluation=combined_feedback,

            model_key=candidate_model,
        )


        row[
            "v2_content"
        ] = v2_content


        row[
            "v2_revision_latency"
        ] = revision_latency


        print(
            f"✅ V2 generated "
            f"in "
            f"{revision_latency:.2f}s"
        )


    except Exception as error:

        print(
            f"❌ Revision failed: "
            f"{error}"
        )


        row[
            "status"
        ] = "FAILED_REVISION"


        row[
            "error"
        ] = str(
            error
        )


        return row


    # =====================================================
    # Evaluate V2
    # =====================================================

    print(
        "\nEvaluating V2 "
        "with all Judges..."
    )


    v2_judges = run_all_judges(
        case_id=case_id,

        challenge=challenge,

        candidate_model=candidate_model,

        version="v2",

        brand_info=brand_info,

        campaign_brief=campaign_brief,

        content=v2_content,

        judge_rows=judge_rows,
    )


    add_judge_scores_to_row(
        row=row,

        version="v2",

        judge_results=v2_judges,
    )


    v2_aggregate = aggregate_judges(
        v2_judges
    )


    row[
        "v2_valid_judge_count"
    ] = v2_aggregate[
        "valid_judge_count"
    ]


    row[
        "v2_expected_judge_count"
    ] = v2_aggregate[
        "expected_judge_count"
    ]


    row[
        "v2_judge_panel_complete"
    ] = v2_aggregate[
        "panel_complete"
    ]


    row[
        "v2_cross_judge_overall"
    ] = v2_aggregate[
        "avg_overall_score"
    ]


    row[
        "v2_cross_judge_claim_risk"
    ] = v2_aggregate[
        "avg_claim_risk"
    ]


    row[
        "v2_judge_score_gap"
    ] = v2_aggregate[
        "judge_score_gap"
    ]


    # =====================================================
    # Stop if V2 Judge Panel is incomplete
    # =====================================================

    if not v2_aggregate[
        "panel_complete"
    ]:

        print(
            "\n❌ V2 Cross-Judge Panel "
            "is incomplete."
        )


        row[
            "overall_improvement"
        ] = None


        row[
            "claim_risk_reduction"
        ] = None


        row[
            "revision_success"
        ] = None


        row[
            "status"
        ] = "FAILED_V2_JUDGES"


        row[
            "error"
        ] = (
            "One or more V2 Judges failed "
            "after all retry attempts."
        )


        return row


    # =====================================================
    # V1 vs V2 Improvement
    # =====================================================

    overall_improvement = round(
        v2_aggregate[
            "avg_overall_score"
        ]

        - v1_aggregate[
            "avg_overall_score"
        ],

        2,
    )


    claim_risk_reduction = round(
        v1_aggregate[
            "avg_claim_risk"
        ]

        - v2_aggregate[
            "avg_claim_risk"
        ],

        2,
    )


    row[
        "overall_improvement"
    ] = overall_improvement


    row[
        "claim_risk_reduction"
    ] = claim_risk_reduction


    # =====================================================
    # Revision Success
    # =====================================================

    revision_success = (
        overall_improvement > 0

        and claim_risk_reduction >= 0
    )


    row[
        "revision_success"
    ] = revision_success


    row[
        "status"
    ] = "SUCCESS_REVISED"


    return row


# =========================================================
# Final Benchmark Summary
# =========================================================

def print_summary(
    batch_df: pd.DataFrame,
    judge_df: pd.DataFrame,
):
    """
    Print Candidate and Cross-Judge summaries.
    """

    print(
        "\n\n"
        + "=" * 80
    )


    print(
        "GROWTHPILOT BENCHMARK SUMMARY"
    )


    print(
        "=" * 80
    )


    successful = batch_df[
        batch_df[
            "status"
        ].str.startswith(
            "SUCCESS",
            na=False,
        )
    ]


    if successful.empty:

        print(
            "\nNo successful experiments."
        )

        return


    # =====================================================
    # Candidate Summary
    # =====================================================

    print(
        "\nCANDIDATE MODEL SUMMARY"
    )


    print(
        "-" * 80
    )


    for candidate_model in CANDIDATE_MODELS:

        candidate_rows = successful[
            successful[
                "candidate_model"
            ]
            == candidate_model
        ]


        if candidate_rows.empty:
            continue


        print(
            f"\nCandidate: "
            f"{candidate_model}"
        )


        print(
            f"Completed cases: "
            f"{len(candidate_rows)}"
        )


        print(
            "Average V1 Cross-Judge Score: "
            f"{candidate_rows['v1_cross_judge_overall'].mean():.2f}"
        )


        print(
            "Average V1 Claim Risk: "
            f"{candidate_rows['v1_cross_judge_claim_risk'].mean():.2f}"
        )


        print(
            "Average Generation Latency: "
            f"{candidate_rows['v1_generation_latency'].mean():.2f}s"
        )


        print(
            "Average Judge Disagreement Gap: "
            f"{candidate_rows['v1_judge_score_gap'].mean():.2f}"
        )


        revision_rows = candidate_rows[
            candidate_rows[
                "revision_triggered"
            ] == True
        ]


        print(
            "Revision Trigger Rate: "
            f"{len(revision_rows) / len(candidate_rows):.1%}"
        )


        if not revision_rows.empty:

            valid_revision_rows = (
                revision_rows[
                    revision_rows[
                        "overall_improvement"
                    ].notna()
                ]
            )


            if not valid_revision_rows.empty:

                print(
                    "Average Revision Improvement: "
                    f"{valid_revision_rows['overall_improvement'].mean():.2f}"
                )


                print(
                    "Average Claim Risk Reduction: "
                    f"{valid_revision_rows['claim_risk_reduction'].mean():.2f}"
                )


    # =====================================================
    # Cross-Judge Analysis
    # =====================================================

    print(
        "\n\nCROSS-JUDGE ANALYSIS — V1"
    )


    print(
        "-" * 80
    )


    v1_judges = judge_df[
        (
            judge_df[
                "version"
            ] == "v1"
        )

        & (
            judge_df[
                "status"
            ] == "SUCCESS"
        )
    ]


    for judge_model in JUDGE_MODELS:

        judge_data = v1_judges[
            v1_judges[
                "judge_model"
            ]
            == judge_model
        ]


        if judge_data.empty:
            continue


        print(
            f"\nJudge: "
            f"{judge_model}"
        )


        for candidate_model in CANDIDATE_MODELS:

            scores = judge_data[
                judge_data[
                    "candidate_model"
                ]
                == candidate_model
            ][
                "overall_score"
            ]


            if scores.empty:
                continue


            print(
                f"  Candidate "
                f"{candidate_model}: "
                f"{scores.mean():.2f}"
            )


    # =====================================================
    # Own-model Score Advantage
    # =====================================================

    print(
        "\n\nOBSERVED OWN-MODEL SCORE ADVANTAGE"
    )


    print(
        "-" * 80
    )


    print(
        "This is an exploratory signal only. "
        "It does NOT prove self-preference bias."
    )


    for judge_model in JUDGE_MODELS:

        judge_data = v1_judges[
            v1_judges[
                "judge_model"
            ]
            == judge_model
        ]


        own_scores = judge_data[
            judge_data[
                "candidate_model"
            ]
            == judge_model
        ][
            "overall_score"
        ]


        other_candidates = [
            model

            for model
            in CANDIDATE_MODELS

            if model != judge_model
        ]


        other_scores = judge_data[
            judge_data[
                "candidate_model"
            ].isin(
                other_candidates
            )
        ][
            "overall_score"
        ]


        if (
            not own_scores.empty
            and not other_scores.empty
        ):

            advantage = (
                own_scores.mean()
                - other_scores.mean()
            )


            print(
                f"\n{judge_model} Judge:"
            )


            print(
                f"Own candidate avg: "
                f"{own_scores.mean():.2f}"
            )


            print(
                f"Other candidate avg: "
                f"{other_scores.mean():.2f}"
            )


            print(
                f"Observed advantage: "
                f"{advantage:+.2f}"
            )


# =========================================================
# Main Experiment
# =========================================================

if __name__ == "__main__":

    cases = load_cases()


    print(
        "\n"
        + "=" * 80
    )


    print(
        "GrowthPilot Cross-Judge Benchmark"
    )


    print(
        "=" * 80
    )


    print(
        f"\nEvaluation Cases: "
        f"{len(cases)}"
    )


    print(
        f"Candidate Models: "
        f"{CANDIDATE_MODELS}"
    )


    print(
        f"Judge Models: "
        f"{JUDGE_MODELS}"
    )


    print(
        f"Judge Max Attempts: "
        f"{JUDGE_MAX_ATTEMPTS}"
    )


    total_candidate_runs = (
        len(cases)
        * len(
            CANDIDATE_MODELS
        )
    )


    print(
        f"Candidate-Case Runs: "
        f"{total_candidate_runs}"
    )


    batch_rows = []

    judge_rows = []


    run_number = 0


    # =====================================================
    # Run All Cases
    # =====================================================

    for case_index, case in enumerate(
        cases,
        start=1,
    ):

        case_id = get_case_id(
            case,
            case_index,
        )


        challenge = get_challenge(
            case
        )


        for candidate_model in CANDIDATE_MODELS:

            run_number += 1


            print(
                "\n\n"
                + "=" * 80
            )


            print(
                f"[{run_number}/"
                f"{total_candidate_runs}] "
                f"{case_id}"
            )


            print(
                f"Candidate: "
                f"{candidate_model}"
            )


            if challenge:

                print(
                    f"Challenge: "
                    f"{challenge}"
                )


            print(
                "=" * 80
            )


            try:

                row = run_candidate_case(
                    case=case,

                    case_index=case_index,

                    candidate_model=(
                        candidate_model
                    ),

                    judge_rows=(
                        judge_rows
                    ),
                )


            except Exception as error:

                print(
                    f"\n❌ Unexpected failure: "
                    f"{error}"
                )


                row = {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "status":
                        "FAILED_UNEXPECTED",

                    "error":
                        str(
                            error
                        ),
                }


            batch_rows.append(
                row
            )


            # =================================================
            # Save Immediately
            # =================================================

            save_results(
                batch_rows,
                judge_rows,
            )


            print(
                "\n💾 Intermediate results saved."
            )


    # =====================================================
    # Final Save
    # =====================================================

    save_results(
        batch_rows,
        judge_rows,
    )


    batch_df = pd.DataFrame(
        batch_rows
    )


    judge_df = pd.DataFrame(
        judge_rows
    )


    # =====================================================
    # Summary
    # =====================================================

    print_summary(
        batch_df,
        judge_df,
    )


    print(
        "\n\n"
        + "=" * 80
    )


    print(
        "Benchmark finished."
    )


    print(
        f"\nMain results:\n"
        f"{BATCH_RESULTS_FILE}"
    )


    print(
        f"\nJudge-level results:\n"
        f"{JUDGE_RESULTS_FILE}"
    )


    print(
        "=" * 80
    )