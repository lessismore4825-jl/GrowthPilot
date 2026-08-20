"""English UI strings for GrowthPilot.

This module contains UI strings only.
Submitted content, evidence, source quotes, policy basis, and model findings
must remain untranslated in app.py.
"""

TEXT = {'judge_disagreement_details': 'Judge Disagreement Details',
 'review_mode_result': 'Review Decision',
 'fast_review_result_message': 'Fast Review used one Judge. This mode prioritizes responsiveness.',
 'cross_judge_decision': 'Cross-Judge Decision',
 'compliance_short': 'Compliance',
 'requirements_short': 'Requirements',
 'advisory_short': 'Advisory',
 'final_routing': 'Final Routing',
 'human_review_required_message': 'The two Judges disagree on at least one mandatory-layer '
                                  'finding. Automatic mandatory revision is disabled. Human review '
                                  'is required.',
 'consensus_compliance_message': 'Both Judges confirmed one or more compliance findings. Mandatory '
                                 'compliance correction is authorized.',
 'consensus_requirement_message': 'Both Judges confirmed one or more missing campaign '
                                  'requirements. Mandatory completion is authorized.',
 'consensus_both_message': 'Both Judges confirmed mandatory compliance and campaign-requirement '
                           'findings.',
 'consensus_clear_message': 'The Cross-Judge review found no mandatory action requiring '
                            'escalation.',
 'review_error_message': 'The review could not be completed.',
 'review_status': 'Review Status',
 'human_review_status_message': 'Mandatory-layer disagreement requires Human Review. No automatic '
                                'clearance has been granted.',
 'human_review_status_explanation': 'A zero consensus-finding count does not mean the content has '
                                    'passed. At least one Judge raised a mandatory concern that '
                                    'was not independently confirmed by the other Judge.',
 'mandatory_detected_message': '{total} mandatory action(s) detected: {compliance} compliance + '
                               '{requirement} requirement.',
 'no_mandatory_correction': 'No mandatory correction or completion was detected by the current '
                            'review.',
 'human_decision_disclaimer': 'AI pre-review is not legal approval and does not replace final '
                              'human publishing judgment.',
 'mandatory_fixes': 'Mandatory Fixes',
 'compliance_issues': 'Compliance Issues',
 'compliance_issue': 'Compliance Issue',
 'problematic_content': 'Problematic Content',
 'policy_source': 'Policy Source',
 'policy_basis': 'Policy Basis',
 'why_conflicts': 'Why It Conflicts',
 'required_action': 'Required Action',
 'no_consensus_compliance': 'No compliance finding was confirmed by both Judges, but an unresolved '
                            'compliance disagreement remains. Human Review is required.',
 'no_compliance_issue': 'No direct, source-grounded blocking conflict detected.',
 'requirement_issues': 'Missing Campaign Requirements',
 'requirement_label': 'Requirement',
 'requirement_id': 'Requirement ID',
 'match_mode': 'Match Mode',
 'why_missing': 'Why It Is Missing',
 'verification': 'Verification',
 'no_consensus_requirement': 'No missing campaign requirement was confirmed by both Judges, but an '
                             'unresolved requirement disagreement remains. Human Review is '
                             'required.',
 'no_requirement_missing': 'No structured Must Mention requirement is missing.',
 'human_review_gate': 'This content has not been automatically cleared. Resolve the Judge '
                      'disagreement before any mandatory AI revision or publishing decision.',
 'optional_improvements': 'Optional Improvements',
 'advisory_count_message': '{count} non-blocking advisory finding(s) were generated.',
 'advisory_label': 'Advisory',
 'relevant_content': 'Relevant Content',
 'why_it_matters': 'Why It Matters',
 'edit_direction': 'Edit Direction',
 'advice_basis': 'Advice Basis',
 'source_quote': 'Source quote',
 'grounding_review_signal': 'Manual review signal',
 'general_heuristic_guidance': 'General AI / marketing guidance',
 'no_optional_improvement': 'No major optional improvement was identified.',
 'diagnostic_signals': 'Diagnostic Signals',
 'heuristic_composite': 'Heuristic Composite',
 'diagnostic_score_help': 'Diagnostic comparison signal only. It is not a calibrated pass/fail '
                          'threshold.',
 'factual_consistency': 'Factual Consistency',
 'unsupported_claim_risk': 'Unsupported Claim Risk',
 'lower_is_better': 'Lower is better.',
 'dimension': 'Dimension',
 'brand_alignment': 'Brand Alignment',
 'tone_match': 'Tone Match',
 'selling_point_coverage': 'Selling Point Coverage',
 'score': 'Score',
 'human_review_notes': 'Human Review Notes',
 'revision_result': 'Revision Result',
 'mandatory_findings': 'Mandatory Findings',
 'compliance_findings': 'Compliance Findings',
 'diagnostic_score_change': 'Diagnostic Score Change',
 'all_mandatory_cleared': 'All previously detected mandatory findings were cleared in the current '
                          're-check.',
 'some_mandatory_remaining': 'Some mandatory findings were removed, but additional attention is '
                             'still required.',
 'mandatory_not_cleared': 'The mandatory revision did not clear all mandatory findings.',
 'app_title': '🚀 GrowthPilot',
 'app_caption': 'AI Campaign Content Copilot for Brand & Agency Teams',
 'app_description': '\n'
                    '**Policy decides what must be corrected.  \n'
                    'Campaign requirements decide what must be completed.  \n'
                    'AI advises what could be improved.  \n'
                    'Humans decide what should ultimately be used.**\n',
 'language_scope_note': 'UI language only. Submitted content, evidence, source quotes and model '
                        'findings are not automatically translated.',
 'model_configuration': 'Model Configuration',
 'generator_reviser_model': 'Generator / Reviser Model',
 'review_mode': 'Review Mode',
 'review_mode_selector': 'Evaluation Strategy',
 'fast_review': '⚡ Fast Review',
 'cross_judge_review': '🛡️ Cross-Judge Review',
 'primary_demo_judge': 'Primary Demo Judge',
 'fast_review_description': 'Fast Review uses one Judge and prioritizes interactive response '
                            'speed.',
 'cross_judge_models': 'Cross-Judge Models',
 'cross_judge_description': 'Two independent Judges evaluate the same content. Only cross-judge '
                            'consensus can authorize automatic mandatory correction. Mandatory '
                            'disagreement is escalated to Human Review.',
 'cross_judge_latency_note': 'Cross-Judge Review may take significantly longer because both models '
                             'must complete evaluation.',
 'campaign_context': 'Campaign Context',
 'brand_product': 'Brand & Product',
 'brand_information': 'Brand Information',
 'verified_product_information': 'Verified Product Information',
 'campaign_brief_requirements': 'Campaign Brief & Requirements',
 'campaign_brief': 'Campaign Brief',
 'platform': 'Platform',
 'content_type': 'Content Type',
 'must_mention': 'Must Mention',
 'must_mention_help': 'One requirement per line. Default is SEMANTIC. Use EXACT | #BrandCampaign '
                      'when exact wording is required.',
 'must_avoid': 'Must Avoid',
 'creator_context': 'Creator Context',
 'creator_category': 'Creator Category',
 'creator_audience': 'Creator Audience',
 'creator_style': 'Creator Style',
 'content_characteristics': 'Content Characteristics',
 'additional_policy_context': 'Additional Policy Context',
 'policy_optional': 'Additional Policy Context (Optional)',
 'policy_placeholder': 'Paste applicable internal brand rules, advertising policy, or platform '
                       'requirements here. Only supplied rules may be used as hard external '
                       'compliance basis.',
 'rag_note': 'A future RAG policy layer can populate this context automatically.',
 'create_tab': '✍️ Create Guidance / Generate Draft',
 'review_tab': '🔎 Review Creator Draft',
 'create_title': 'Create Guidance / Generate Draft',
 'create_description': 'Generate creator-specific campaign guidance and a reference draft before '
                       'content production.',
 'none_specified': 'None specified.',
 'generate_guidance_button': 'Generate Guidance & Reference Draft',
 'missing_brand': 'Please provide Brand Information.',
 'missing_product': 'Please provide Verified Product Information.',
 'missing_campaign': 'Please provide a Campaign Brief.',
 'creating_guidance': 'Creating guidance with {model}...',
 'generating_reference_draft': 'Generating reference draft with {model}...',
 'guidance_generation_failed': 'Guidance or draft generation failed: {error}',
 'creator_guidance': 'Creator Guidance',
 'guidance_advisory_note': 'AI-generated guidance is advisory. Structured Must Mention / Must '
                           'Avoid inputs remain authoritative.',
 'reference_draft': 'Reference Draft',
 'generated_reference_draft': 'Generated Reference Draft',
 'guidance_latency': 'Guidance Latency',
 'draft_generation_latency': 'Draft Generation Latency',
 'review_reference_draft': 'Review Reference Draft',
 'running_review': 'Running content review...',
 'reference_review_failed': 'Reference draft review failed: {error}',
 'reference_draft_review': 'Reference Draft Review',
 'review_latency': 'Review Latency',
 'actions': 'Actions',
 'automatic_revision_disabled': 'Automatic mandatory revision is disabled because the Judges '
                                'disagree. Resolve the disagreement through Human Review first.',
 'mandatory_before_optional': 'Cross-Judge consensus authorized mandatory correction. Mandatory '
                              'findings should be resolved before optional optimization.',
 'mandatory_fix': 'Apply Mandatory Fix',
 'applying_mandatory_revision': 'Applying the minimum mandatory revision...',
 'rechecking_revised_draft': 'Re-checking the revised draft using the same review mode...',
 'mandatory_revision_failed': 'Mandatory revision failed: {error}',
 'optional_optimization_available': 'No mandatory action is required. Quality optimization remains '
                                    'optional and must be triggered by the user.',
 'quality_optimization': 'Optimize Quality (Optional)',
 'applying_optional_improvements': 'Applying optional advisory improvements...',
 'reviewing_optimized_content': 'Reviewing optimized content using the same review mode...',
 'quality_optimization_failed': 'Quality optimization failed: {error}',
 'no_mandatory_issue': 'No mandatory issue or major advisory was identified.',
 'revised_reference_draft': 'Revised Reference Draft',
 'revision_mode': 'Revision Mode',
 'revised_content': 'Revised Content',
 'revision_latency': 'Revision Latency',
 'recheck_latency': 'Re-check Latency',
 'revised_draft_review': 'Revised Draft Review',
 'review_creator_title': 'Review Creator Draft',
 'review_creator_description': 'Paste a creator-returned draft for pre-publication review. '
                               'First-person creator language is treated as creator-authored '
                               'context, while factual claims and supplied campaign rules are '
                               'reviewed normally.',
 'creator_draft': 'Creator Draft',
 'review_creator_button': 'Review Creator Draft',
 'missing_creator_draft': 'Please paste a Creator Draft.',
 'creator_review_failed': 'Creator Draft review failed: {error}',
 'creator_draft_review': 'Creator Draft Review',
 'creator_mandatory_warning': 'Cross-Judge consensus confirmed mandatory findings. The revision '
                              'below applies only the minimum mandatory changes and still requires '
                              'final human approval.',
 'rechecking_creator_draft': 'Re-checking revised Creator Draft using the same review mode...',
 'creator_optional_optimization': 'No mandatory action is required. Advisory optimization remains '
                                  'optional and is only applied when triggered by the user.',
 'reviewing_optimized_creator_draft': 'Reviewing optimized Creator Draft using the same review '
                                      'mode...',
 'revised_creator_draft': 'AI-Assisted Revised Creator Draft',
 'revised_creator_review': 'Revised Creator Draft Review',
 'footer': 'GrowthPilot MVP v2 · Creator-Specific Guidance + Cross-Judge Policy Review + '
           'Human-in-the-Loop Approval'}

TRANSLATIONS = TEXT
EN_TEXT = TEXT
ENGLISH_TEXT = TEXT