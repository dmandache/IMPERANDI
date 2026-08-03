import pandas as pd
import pytest
import yaml

from imperandi.annotations import (
    apply_ontologies,
    apply_rule_packs,
    resolve_annotation,
)
from imperandi.config.models import OntologyConfig
from imperandi.pipeline.stages.core import _copy_builtin_annotations, _phase_to_slot
from imperandi.pipeline.stages.imaging import _canonical_image_phase


def test_composite_ontology_can_populate_clinical_slot(tmp_path):
    ontology_path = tmp_path / "protocols.csv"
    pd.DataFrame(
        {
            "SeriesDescription": ["Foie mixte", "Foie mixte"],
            "AcquisitionNumber": [1, 2],
            "clinical_slot": ["CT_ARTERIAL", "CT_PORTAL_VENOUS"],
        }
    ).to_csv(ontology_path, index=False)
    config = OntologyConfig.model_validate(
        {
            "id": "site_protocols",
            "source": ontology_path,
            "keys": {
                "SeriesDescription": {"match": "normalized_exact"},
                "AcquisitionNumber": {"match": "numeric_exact"},
            },
            "output": {
                "value_column": "clinical_slot",
                "target_column": "slot_ontology",
                "vocabulary": "clinical_slot",
            },
        }
    )
    data = pd.DataFrame(
        {"SeriesDescription": [" FOIE  MIXTE "], "AcquisitionNumber": [2]}
    )
    result = apply_ontologies(data, [config])
    assert result.loc[0, "slot_ontology"] == "CT_PORTAL_VENOUS"
    assert result.loc[0, "slot_ontology_ontology_id"] == "site_protocols"


def test_ontology_unwraps_singleton_aggregated_key_values(tmp_path):
    ontology_path = tmp_path / "protocols.csv"
    pd.DataFrame(
        {"SeriesDescription": ["LIVER PORTAL"], "clinical_slot": ["CT_PORTAL_VENOUS"]}
    ).to_csv(ontology_path, index=False)
    config = OntologyConfig.model_validate(
        {
            "id": "singleton",
            "source": ontology_path,
            "keys": {"SeriesDescription": {"match": "normalized_exact"}},
            "output": {
                "value_column": "clinical_slot",
                "target_column": "slot_ontology",
                "vocabulary": "clinical_slot",
            },
        }
    )

    result = apply_ontologies(
        pd.DataFrame({"SeriesDescription": [[" liver portal "]]}), [config]
    )

    assert result.loc[0, "slot_ontology"] == "CT_PORTAL_VENOUS"


def test_ontology_supports_arbitrary_derived_column(tmp_path):
    path = tmp_path / "families.csv"
    pd.DataFrame(
        {"ProtocolName": ["LIVER DYNAMIC"], "family": ["liver_dynamic"]}
    ).to_csv(path, index=False)
    config = OntologyConfig.model_validate(
        {
            "id": "families",
            "source": path,
            "keys": {"ProtocolName": {"match": "normalized_exact"}},
            "output": {"value_column": "family", "target_column": "protocol_family"},
        }
    )
    result = apply_ontologies(
        pd.DataFrame({"ProtocolName": ["liver dynamic"]}), [config]
    )
    assert result.loc[0, "protocol_family"] == "liver_dynamic"


def test_ontology_can_flag_conflicting_composite_keys(tmp_path):
    path = tmp_path / "conflicting.csv"
    pd.DataFrame(
        {
            "ProtocolName": ["LIVER", "LIVER"],
            "family": ["dynamic", "routine"],
        }
    ).to_csv(path, index=False)
    config = OntologyConfig.model_validate(
        {
            "id": "conflicting",
            "source": path,
            "keys": {"ProtocolName": {"match": "normalized_exact"}},
            "output": {"value_column": "family", "target_column": "family"},
            "conflicts": "flag",
        }
    )

    result = apply_ontologies(pd.DataFrame({"ProtocolName": ["liver"]}), [config])

    assert result.loc[0, "family"] == "dynamic"
    assert bool(result.loc[0, "family_conflict"])


def test_later_unmatched_ontology_preserves_existing_value_and_provenance(tmp_path):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    pd.DataFrame({"ProtocolName": ["LIVER"], "family": ["dynamic"]}).to_csv(
        first_path, index=False
    )
    pd.DataFrame({"ProtocolName": ["KIDNEY"], "family": ["renal"]}).to_csv(
        second_path, index=False
    )
    configs = [
        OntologyConfig.model_validate(
            {
                "id": ontology_id,
                "source": source,
                "keys": {"ProtocolName": {"match": "normalized_exact"}},
                "output": {
                    "value_column": "family",
                    "target_column": "protocol_family",
                },
            }
        )
        for ontology_id, source in [("first", first_path), ("second", second_path)]
    ]

    result = apply_ontologies(pd.DataFrame({"ProtocolName": ["liver"]}), configs)

    assert result.loc[0, "protocol_family"] == "dynamic"
    assert result.loc[0, "protocol_family_ontology_id"] == "first"


def test_rules_can_set_values_and_exclude(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "phase.portal",
                        "target": "phase_rules_explicit",
                        "value": "PORTAL_VENOUS",
                        "priority": 100,
                        "when": {
                            "any": [
                                {
                                    "column": "SeriesDescription",
                                    "operator": "regex",
                                    "value": "portal|veineux",
                                }
                            ]
                        },
                    },
                    {
                        "id": "exclude.scout",
                        "action": "exclude",
                        "reason": "localizer",
                        "priority": 200,
                        "when": {
                            "any": [
                                {
                                    "column": "SeriesDescription",
                                    "operator": "contains",
                                    "value": "scout",
                                }
                            ]
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = apply_rule_packs(
        pd.DataFrame({"SeriesDescription": ["Foie portal", "Scout abdomen"]}),
        [str(path)],
    )
    assert result.loc[0, "phase_rules_explicit"] == "PORTAL_VENOUS"
    assert bool(result.loc[1, "eligible"]) is False
    assert result.loc[1, "exclusion_rule_id"] == "exclude.scout"


def test_rule_priorities_apply_across_multiple_packs(tmp_path):
    high = tmp_path / "high.yaml"
    low = tmp_path / "low.yaml"
    shared_condition = {"any": [{"column": "ProtocolName", "operator": "exists"}]}
    high.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "high",
                        "target": "protocol_family",
                        "value": "high-priority",
                        "priority": 100,
                        "when": shared_condition,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    low.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "low",
                        "target": "protocol_family",
                        "value": "low-priority",
                        "priority": 1,
                        "when": shared_condition,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = apply_rule_packs(
        pd.DataFrame({"ProtocolName": ["LIVER"]}), [str(high), str(low)]
    )

    assert result.loc[0, "protocol_family"] == "high-priority"
    assert result.loc[0, "protocol_family_rule_id"] == "high"


def test_eq_rule_treats_nullable_metadata_as_not_matching(tmp_path):
    path = tmp_path / "nullable.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "ct-only",
                        "target": "route",
                        "value": "CT",
                        "when": {
                            "any": [
                                {
                                    "column": "Modality",
                                    "operator": "eq",
                                    "value": "CT",
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = apply_rule_packs(
        pd.DataFrame({"Modality": [pd.NA, None, "CT", ["MR", "CT"]]}),
        [str(path)],
    )

    assert result["route"].isna().tolist() == [True, True, False, False]


def test_equal_priority_exclusion_conflicts_are_rejected(tmp_path):
    paths = []
    for rule_id, reason in [("exclude-a", "reason-a"), ("exclude-b", "reason-b")]:
        path = tmp_path / f"{rule_id}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": rule_id,
                            "action": "exclude",
                            "reason": reason,
                            "priority": 10,
                            "when": {
                                "any": [
                                    {
                                        "column": "Modality",
                                        "operator": "exists",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(str(path))

    with pytest.raises(ValueError, match="equal priority"):
        apply_rule_packs(pd.DataFrame({"Modality": ["CT"]}), paths)


def test_rules_reject_invalid_controlled_evidence_during_load(tmp_path):
    path = tmp_path / "invalid-slot.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "invalid-slot",
                        "target": "slot_rules_explicit",
                        "value": "CT_MADE_UP",
                        "when": {"any": [{"column": "Modality", "operator": "exists"}]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="controlled target"):
        apply_rule_packs(pd.DataFrame({"Modality": ["CT"]}), [str(path)])


def test_rules_reject_invalid_regular_expressions_during_load(tmp_path):
    path = tmp_path / "invalid-regex.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "invalid-regex",
                        "target": "protocol_family",
                        "value": "dynamic",
                        "when": {
                            "any": [
                                {
                                    "column": "SeriesDescription",
                                    "operator": "regex",
                                    "value": "[unterminated",
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="regular expression"):
        apply_rule_packs(
            pd.DataFrame({"SeriesDescription": ["dynamic"]}),
            [str(path)],
        )


def test_rule_pack_requires_an_explicit_schema_version(tmp_path):
    path = tmp_path / "unversioned.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "id": "family",
                        "target": "protocol_family",
                        "value": "dynamic",
                        "when": {"any": [{"column": "Modality", "operator": "exists"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version"):
        apply_rule_packs(pd.DataFrame({"Modality": ["CT"]}), [str(path)])


def test_malformed_rule_pack_reports_its_path(tmp_path):
    path = tmp_path / "broken-rules.yaml"
    path.write_text("rules: [unterminated", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken-rules\.yaml"):
        apply_rule_packs(pd.DataFrame({"Modality": ["CT"]}), [str(path)])


def test_annotation_resolution_preserves_sources_and_flags_disagreement():
    data = pd.DataFrame(
        {
            "phase_ontology": ["PORTAL_VENOUS", pd.NA],
            "phase_rules_explicit": ["ARTERIAL", "ARTERIAL"],
            "phase_image": ["ARTERIAL", "PORTAL_VENOUS"],
        }
    )
    result = resolve_annotation(
        data,
        candidates=["phase_ontology", "phase_rules_explicit", "phase_image"],
        target="phase_resolved",
    )
    assert result.loc[0, "phase_resolved"] == "PORTAL_VENOUS"
    assert result.loc[0, "phase_resolved_source"] == "phase_ontology"
    assert bool(result.loc[0, "phase_resolved_conflict"])
    assert result.loc[1, "phase_resolved"] == "ARTERIAL"


def test_custom_rule_evidence_is_not_overwritten_by_builtin_curation():
    annotated = pd.DataFrame(
        {
            "curation_modality": ["CT"],
            "ct_phase": ["PORTAL_VENOUS"],
            "selection_slot": ["CT_PORTAL_VENOUS"],
            "phase_rules_explicit": ["DELAYED"],
            "slot_rules_explicit": ["CT_DELAYED"],
        }
    )

    result = _copy_builtin_annotations(annotated)

    assert result.loc[0, "phase_rules_explicit"] == "DELAYED"
    assert result.loc[0, "slot_rules_explicit"] == "CT_DELAYED"


def test_slot_image_is_derived_only_from_image_phase_evidence():
    ontology_phase = pd.Series(
        {
            "phase_resolved": "PORTAL_VENOUS",
            "phase_source": "phase_ontology",
            "curation_modality": "CT",
        }
    )
    image_phase = ontology_phase.copy()
    image_phase["phase_source"] = "phase_image"

    assert _phase_to_slot(ontology_phase) is None
    assert _phase_to_slot(image_phase) == "CT_PORTAL_VENOUS"


def test_image_phase_normalization_accepts_known_variants_only():
    assert _canonical_image_phase("portal-venous") == "PORTAL_VENOUS"
    assert _canonical_image_phase("non contrast") == "NATIVE"
    assert pd.isna(_canonical_image_phase("unexpected-new-label"))
