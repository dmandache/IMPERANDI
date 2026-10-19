"""Public rule extensions must change real curation and remain explainable."""

from dataclasses import dataclass

import pandas as pd
import pytest

from imperandi.curation import (
    CurationEngine,
    CurationPolicy,
    RuleContext,
    RuleResult,
    curate_by_modality,
    curate_ct,
    curate_mri,
)
from imperandi.curation.phase_resolution import (
    phase_curation_input_columns,
    validate_phase_curation,
)
from imperandi.curation.policy import BUILTIN_RULES


def cohort(modality="MR", description="AX T1 VIBE dynamic", count=4):
    return pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "s1",
                "series_id": "ser1",
                "volume_id": f"v{i}",
                "date": "2020-01-01",
                "Modality": modality,
                "SeriesDescription": description,
                "ProtocolName": "site_protocol",
                "ImageType": "ORIGINAL PRIMARY AXIAL",
                "time": f"120{i}00",
                "acquisition_order": i,
                "Rows": 512,
                "Columns": 512,
                "n_files": 120,
                "SliceThickness": 3.0,
                "ReviewPhase": "",
            }
            for i in range(count)
        ]
    )


@dataclass(frozen=True)
class ReviewRule:
    name: str = "site_review"
    modality: str = "MR"
    version: str = "1"
    input_columns: tuple[str, ...] = ("ReviewPhase",)
    requires: tuple[str, ...] = ()

    def evaluate(self, context: RuleContext):
        for idx, row in context.rows.iterrows():
            yield RuleResult(idx, row.ReviewPhase or None, "site reviewer phase")


@pytest.mark.parametrize(
    "modality,description", [("CT", "AX portal"), ("MR", "AX T1 VIBE dynamic")]
)
def test_default_engine_matches_legacy_api(modality, description):
    data = cohort(modality, description)
    expected = curate_by_modality(data)
    actual = CurationEngine().curate(data)
    pd.testing.assert_frame_equal(actual.curated, expected["curated_all"])
    pd.testing.assert_frame_equal(actual.selected_long, expected["selected_long_all"])
    assert actual.explain("v0").iloc[-1].disposition == "final"
    pd.testing.assert_frame_equal(data, cohort(modality, description))


@pytest.mark.parametrize("modality", ["MR", "CT"])
def test_custom_rule_changes_phase_and_selected_slot(modality):
    data = cohort(modality, "AX T1 VIBE" if modality == "MR" else "AX abdomen", count=1)
    data["ReviewPhase"] = "PORTAL_VENOUS"
    policy = CurationPolicy.from_manifest().with_rule(
        ReviewRule(modality=modality),
        precedes="generic_dynamic" if modality == "MR" else "pre_post_native",
    )
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist() == ["PORTAL_VENOUS"]
    assert result.selected_long.selection_slot.tolist() == [
        "T1_PORTAL_VENOUS" if modality == "MR" else "CT_PORTAL_VENOUS"
    ]
    decision = result.explain("v0").query("rule == 'site_review'").iloc[0]
    assert decision.disposition == "accepted"
    assert decision.context_row_ids == (0,)


def test_custom_native_is_an_anchor_for_later_inference():
    data = cohort()
    data.loc[0, "ReviewPhase"] = "NATIVE"
    policy = CurationPolicy.from_manifest().with_rule(
        ReviewRule(), precedes="native_context"
    )
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist().count("NATIVE") == 1
    assert result.curated.iloc[0].mri_perfusion_source == "site_review"
    assert result.curated.iloc[1:].phase.ne("OTHER").all()


def test_metadata_precedence_rejects_custom_proposal_with_explanation():
    data = cohort(description="AX T1 VIBE arterial", count=1)
    data.ReviewPhase = "PORTAL_VENOUS"
    policy = CurationPolicy.from_manifest().with_rule(ReviewRule())
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist() == ["ARTERIAL"]
    rejected = result.explain("v0").query("rule == 'site_review'").iloc[0]
    assert rejected.phase == "PORTAL_VENOUS"
    assert rejected.disposition == "rejected"
    assert rejected.detail == "lower precedence than metadata"


def test_precedence_and_execution_dependencies_are_distinct():
    data = cohort("CT", "pre/post", count=2)
    data.ReviewPhase = ["PORTAL_VENOUS", ""]
    # Pre/post runs first, but reviewer evidence has higher precedence.
    policy = CurationPolicy.from_manifest().with_rule(
        ReviewRule(modality="CT", requires=("pre_post_native",)),
        precedes="pre_post_native",
    )
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist()[0] == "PORTAL_VENOUS"
    trace = result.explain("v0")
    assert trace.rule.tolist().index("pre_post_native") < trace.rule.tolist().index(
        "site_review"
    )
    assert trace.query("rule == 'pre_post_native'").iloc[0].disposition == "superseded"


def test_configured_precedence_can_promote_custom_evidence_over_metadata():
    data = cohort(description="AX T1 VIBE arterial", count=1)
    data.ReviewPhase = "PORTAL_VENOUS"
    order = [name for name in BUILTIN_RULES["MR"] if name != "metadata"] + ["metadata"]
    policy = (
        CurationPolicy.from_manifest()
        .with_rule_options(modality="MR", precedence=order)
        .with_rule(ReviewRule())
    )
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist() == ["PORTAL_VENOUS"]


def test_authoritative_strategy_remains_ahead_of_custom_rules():
    data = cohort(count=1)
    data.ReviewPhase = "PORTAL_VENOUS"
    data["OntologyPhase"] = "native"
    policy = CurationPolicy.from_manifest(
        {
            "phase_curation": {
                "strategies": [
                    {
                        "type": "ontology",
                        "columns": ["OntologyPhase"],
                        "mapping": {"native": "NATIVE"},
                    },
                    {"type": "rules"},
                ]
            }
        }
    ).with_rule(ReviewRule())
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist() == ["NATIVE"]
    assert (
        "authoritative"
        in result.explain("v0").query("rule == 'site_review'").iloc[0].detail
    )


@pytest.mark.parametrize(
    "modality,description", [("CT", "localizer"), ("MR", "AX T2 HASTE")]
)
def test_custom_rules_cannot_override_phase_applicability(modality, description):
    data = cohort(modality, description, count=1)
    data.ReviewPhase = "ARTERIAL"
    policy = CurationPolicy.from_manifest().with_rule(
        ReviewRule(modality=modality), precedes="pre_post_native"
    )
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase_status.tolist() == ["NOT_APPLICABLE"]
    assert (
        result.explain("v0").query("rule == 'site_review'").iloc[0].disposition
        == "rejected"
    )


@pytest.mark.parametrize("scope", ["row", "exam", "compatible_protocol"])
def test_custom_context_respects_patient_and_study_boundaries(scope):
    data = cohort(count=4)
    data.patient_key = ["p1", "p1", "p2", "p1"]
    data.study_id = ["s1", "s1", "s1", "s2"]
    policy = CurationPolicy.from_manifest(
        {
            "phase_curation": {
                "strategies": [{"type": "rules"}],
                "best_candidate_group_columns": ["date"],
            }
        }
    ).with_rule(ReviewRule(), context=scope)
    trace = CurationEngine(policy).curate(data).trace.query("rule == 'site_review'")
    for ids in trace.context_row_ids:
        assert data.loc[list(ids), "patient_key"].nunique() == 1
        assert data.loc[list(ids), "study_id"].nunique() == 1
        if scope == "row":
            assert len(ids) == 1


def test_compatible_protocol_separates_mri_families():
    data = cohort(count=2)
    data.SeriesDescription = ["AX T1 VIBE dynamic", "AX T1 LAVA dynamic"]
    policy = CurationPolicy.from_manifest().with_rule(ReviewRule())
    trace = CurationEngine(policy).curate(data).trace.query("rule == 'site_review'")
    assert trace.context_row_ids.tolist() == [(0,), (1,)]


def test_context_columns_partition_inference_without_changing_selection_groups():
    data = cohort("CT", "pre/post", count=2)
    data["ReviewGroup"] = ["a", "b"]
    policy = CurationPolicy.from_manifest().with_rule_options(
        modality="CT", context_columns=["ReviewGroup"]
    )
    assert curate_ct(data)["curated"].phase.tolist() == ["NATIVE", "OTHER"]
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase.tolist() == ["OTHER", "OTHER"]
    assert "ReviewGroup" in policy.input_columns
    assert policy.phase_curation["best_candidate_group_columns"] is None


def test_manifest_can_disable_named_rule():
    data = cohort("CT", "pre/post", count=2)
    config = {
        "strategies": [{"type": "rules"}],
        "rule_policy": {"CT": {"disabled": ["pre_post_native"]}},
    }
    assert curate_ct(data, phase_curation=config)["curated"].phase.tolist() == [
        "OTHER",
        "OTHER",
    ]


def test_text_columns_override_and_no_cross_call_leakage():
    data = cohort(description="AX T1 VIBE arterial", count=1)
    data.ProtocolName = "portal"
    policy = CurationPolicy.from_manifest().with_text_columns(
        modality="MR", columns=["ProtocolName", "SeriesDescription"]
    )
    assert CurationEngine(policy).curate(data).curated.phase.tolist() == [
        "PORTAL_VENOUS"
    ]
    assert CurationEngine().curate(data).curated.phase.tolist() == ["ARTERIAL"]
    assert curate_mri(data, curation_policy=policy)["curated"].phase.tolist() == [
        "PORTAL_VENOUS"
    ]


def test_policy_is_defensive_and_fingerprints_changes():
    original = CurationPolicy.from_manifest()
    changed = original.with_rule(ReviewRule())
    config = changed.phase_curation
    config["strategies"].clear()
    assert changed.phase_curation["strategies"]
    assert original.registrations == ()
    assert "ReviewPhase" in changed.input_columns
    assert original.fingerprint != changed.fingerprint
    assert changed.fingerprint == original.with_rule(ReviewRule()).fingerprint
    assert (
        changed.fingerprint != original.with_rule(ReviewRule(version="2")).fingerprint
    )
    assert (
        original.with_rule(ReviewRule(), context="row").fingerprint
        != changed.fingerprint
    )


@pytest.mark.parametrize(
    "options",
    [
        {"disabled": ["metadata"]},
        {"disabled": ["typo"]},
        {"precedence": ["generic_dynamic"]},
        {"context_columns": ["x", "x"]},
        {"typo": True},
    ],
)
def test_invalid_manifest_rule_options_fail(options):
    with pytest.raises(ValueError):
        validate_phase_curation(
            {"strategies": [{"type": "rules"}], "rule_policy": {"MR": options}}
        )


def test_dependency_on_disabled_rule_fails():
    policy = CurationPolicy.from_manifest().with_rule_options(
        modality="MR", disabled=["native_context"]
    )
    policy = policy.with_rule(ReviewRule(requires=("native_context",)))
    with pytest.raises(ValueError, match="dependencies"):
        CurationEngine(policy).curate(cohort())


def test_unknown_duplicate_rules_and_dependencies_fail():
    policy = CurationPolicy.from_manifest()
    with pytest.raises(ValueError, match="precedes"):
        policy.with_rule(ReviewRule(), precedes="typo")
    with pytest.raises(ValueError, match="dependencies"):
        policy.with_rule(ReviewRule(requires=("typo",)))
    with pytest.raises(ValueError, match="Duplicate"):
        policy.with_rule(ReviewRule()).with_rule(ReviewRule())


def test_missing_context_and_custom_input_columns_fail():
    with pytest.raises(ValueError, match="Missing curation context column"):
        CurationEngine(
            CurationPolicy.from_manifest().with_rule_options(
                modality="MR", context_columns=["missing"]
            )
        ).curate(cohort())
    with pytest.raises(ValueError, match="needs input columns"):
        CurationEngine(CurationPolicy.from_manifest().with_rule(ReviewRule())).curate(
            cohort().drop(columns="ReviewPhase")
        )


def test_out_of_context_proposals_fail():
    class InvalidRule(ReviewRule):
        def evaluate(self, context):
            yield RuleResult(999, "NATIVE", "wrong context")

    with pytest.raises(ValueError, match="outside its context"):
        CurationEngine(CurationPolicy.from_manifest().with_rule(InvalidRule())).curate(
            cohort()
        )


def test_duplicate_input_indices_and_volume_ids_have_unambiguous_row_explanations():
    data = pd.concat([cohort(count=1), cohort(count=1)])
    result = CurationEngine().curate(data)
    with pytest.raises(ValueError, match="ambiguous"):
        result.explain("v0")
    assert result.explain(row_id=0).row_id.unique().tolist() == [0]
    assert result.explain(row_id=1).row_id.unique().tolist() == [1]


def test_empty_and_unsupported_modalities():
    assert CurationEngine().curate(pd.DataFrame()).curated.empty
    data = cohort("US")
    result = CurationEngine().curate(data)
    assert result.curated.empty
    pd.testing.assert_frame_equal(result.by_modality["other"], data)


def test_manifest_context_columns_participate_in_resume_inputs():
    config = {
        "strategies": [{"type": "rules"}],
        "rule_policy": {"MR": {"context_columns": ["SiteProtocol"]}},
    }
    assert "SiteProtocol" in phase_curation_input_columns(config)


@pytest.mark.parametrize("unresolved", ["OTHER", "UNKNOWN", "UNCLASSIFIED"])
def test_unresolved_custom_proposals_do_not_block_builtin_fallbacks(unresolved):
    data = cohort()
    data.ReviewPhase = unresolved
    policy = CurationPolicy.from_manifest().with_rule(
        ReviewRule(), precedes="native_context"
    )
    expected = CurationEngine().curate(data)
    actual = CurationEngine(policy).curate(data)
    pd.testing.assert_frame_equal(actual.curated, expected.curated)
    assert actual.trace.query("rule == 'site_review'").disposition.eq("abstained").all()


def test_custom_phase_is_normalized_before_it_enters_context():
    data = cohort(count=1)
    data.ReviewPhase = "portal venous"
    policy = CurationPolicy.from_manifest().with_rule(ReviewRule())
    actual = CurationEngine(policy).curate(data)
    assert actual.curated.mri_perfusion_label.tolist() == ["PORTAL_VENOUS"]


def test_unrelated_modality_does_not_require_custom_inputs():
    data = cohort("CT", "AX portal", count=1).drop(columns="ReviewPhase")
    policy = CurationPolicy.from_manifest().with_rule(ReviewRule())
    assert CurationEngine(policy).curate(data).curated.phase.tolist() == [
        "PORTAL_VENOUS"
    ]


def test_legacy_entry_point_validates_custom_inputs():
    policy = CurationPolicy.from_manifest().with_rule(ReviewRule())
    with pytest.raises(ValueError, match="needs input columns"):
        curate_mri(cohort().drop(columns="ReviewPhase"), curation_policy=policy)


def test_custom_proposal_cannot_hide_structured_dicom_conflict():
    data = cohort(count=1)
    data["ContrastBolusAgentPhase"] = [["ARTERIAL", "PORTAL_VENOUS"]]
    data.ReviewPhase = "ARTERIAL"
    policy = CurationPolicy.from_manifest().with_rule(ReviewRule())
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase_status.tolist() == ["UNRESOLVED"]
    assert (
        "conflicting structured"
        in result.explain("v0").query("rule == 'site_review'").iloc[0].detail
    )


def test_manifest_controls_reach_clean_and_phase_annotation():
    from imperandi.ingest.clean import run_clean_pipeline, validate_cleaning_manifest
    from imperandi.extract.phase import _ensure_modality_curation

    data = cohort("CT", "pre/post", count=2)
    config = validate_phase_curation(
        {
            "strategies": [{"type": "rules"}],
            "rule_policy": {"CT": {"disabled": ["pre_post_native"]}},
        }
    )
    steps = validate_cleaning_manifest(
        {
            "phase_curation": config,
            "cleaning": {
                "steps": [{"type": "modality_curation"}, {"type": "phase_curation"}]
            },
        }
    )
    cleaned = run_clean_pipeline(data, steps, phase_curation=config)
    assert cleaned.phase.tolist() == ["OTHER", "OTHER"]
    annotated = _ensure_modality_curation(data, config)
    assert annotated.rule_phase.tolist() == ["OTHER", "OTHER"]


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_custom_numeric_confidence_and_input_mutation_isolation(modality):
    class NumericRule(ReviewRule):
        def evaluate(self, context):
            # Accidental scalar edits to the snapshot must not alter the cohort.
            context.rows.loc[:, "ReviewPhase"] = "edited snapshot"
            yield RuleResult(context.rows.index[0], "PORTAL_VENOUS", "site model", 0.9)

    data = cohort(modality, "AX T1 VIBE" if modality == "MR" else "AX abdomen", count=1)
    policy = CurationPolicy.from_manifest().with_rule(
        NumericRule(modality=modality), precedes="pre_post_native"
    )
    result = CurationEngine(policy).curate(data)
    assert result.curated.phase_confidence.tolist() == [0.9]
    assert result.curated.ReviewPhase.tolist() == [""]
    assert data.ReviewPhase.tolist() == [""]
