# Curation policies and custom rules

`CurationEngine` provides a Python interface to CT and MRI curation with explicit
rule context and decision explanations. Existing `curate_ct()`, `curate_mri()`,
and `curate_by_modality()` functions remain available with their existing return
dictionaries. Default classification, phase strategies, scores, and selection
policies are preserved.

## Basic use

```python
from imperandi.curation import CurationEngine, CurationPolicy

policy = CurationPolicy.from_manifest()  # Built-in defaults
result = CurationEngine(policy).curate(df)

curated = result.curated
selected = result.selected_long
explanation = result.explain(volume_id="v12")
```

`from_manifest(manifest)` accepts a full manifest mapping, such as the result of
`yaml.safe_load()`. For an existing `phase_curation` mapping, use
`CurationPolicy.from_phase_config(config)`. Policy methods return new policies:

```python
policy = policy.with_text_columns(
    modality="MR",
    columns=["ProtocolName", "SeriesDescription"],
)
policy = policy.with_rule_options(
    modality="MR",
    disabled=["native_context"],
    context_columns=["SiteProtocol"],
)
```

Additional context columns partition inference groups. They do not redefine
`best_candidate_group_columns`, acquisition identity, or MRI protocol compatibility.
Missing configured context columns raise an error. Disabling a rule disables that
specific contextual pass; native labels may still come from explicit metadata,
another inference rule, or a higher-priority phase strategy.

The legacy entry points also accept `curation_policy=policy`. Supplying both a
policy and a conflicting `phase_curation` mapping raises an error.

## Add a Python rule

A rule declares a stable name, modality, version, input columns, execution
dependencies, and an `evaluate(context)` method. No base class is required.
Keep rule objects immutable and free of per-run state so one policy can be reused.

```python
from dataclasses import dataclass
from imperandi.curation import RuleContext, RuleResult


@dataclass(frozen=True)
class SiteSpecificPhaseRule:
    name: str = "site_review"
    modality: str = "MR"
    version: str = "1"
    input_columns: tuple[str, ...] = ("ReviewedPhase",)
    requires: tuple[str, ...] = ()

    def evaluate(self, context: RuleContext):
        for row_id, row in context.rows.iterrows():
            phase = row.get("ReviewedPhase")
            if isinstance(phase, str) and phase.strip():
                yield RuleResult(
                    row_id=row_id,
                    phase=phase,
                    reason="Phase supplied by the site review workflow",
                    confidence="reviewed",
                )


policy = policy.with_rule(
    SiteSpecificPhaseRule(),
    context="compatible_protocol",
    precedes="generic_dynamic",
)
result = CurationEngine(policy).curate(df)
```

Use the dataframe index supplied in `context.rows` as `RuleResult.row_id`. The
engine assigns zero-based input positions internally, so duplicate input indices
do not confuse rule application. A rule cannot propose rows outside its context
or propose the same row twice. Input columns declared by a registered rule must
be present in its modality's input.

Rules return proposals instead of editing the source dataframe. Returning no
proposal, or `phase=None`, means the rule abstains. `OTHER` is unresolved evidence;
custom proposals using `OTHER` or a configured unresolved label also abstain,
preserving subsequent fallback opportunities. The built-in contextual rules may
retract provisional volume-order hints when the wider acquisition context shows
they were unsupported.

`RuleContext` contains a dataframe snapshot, modality, ordered text columns,
grouping columns, scope, and authoritative row identities. Available scopes are:

| Scope | Rows visible to the rule |
| --- | --- |
| `row` | One acquisition row. |
| `exam` | The current inference group, further bounded by available patient/date/study identifiers. |
| `compatible_protocol` | An exam context further partitioned by MRI's existing protocol key, including family, plane, image type, and Dixon component. CT uses available `ProtocolName` and `acquisition_plane` fields. |

Custom rules receive exact compatible groups; they do not automatically inherit
the special missing-component allowances used internally by some MRI algorithms.
Additional `context_columns` narrow every contextual rule in the modality. A
custom rule can also filter `context.rows` to implement site-specific conditions.

## Precedence and execution dependencies

Three distinct orders remain explicit:

1. `phase_curation.strategies` decides which phase strategy wins. Strategies ahead
   of `rules` provide authoritative context. Custom phase rules participate inside
   the `rules` strategy and cannot replace those authoritative decisions.
2. `text_columns` selects the metadata field supplying a decision. Structured
   metadata continues to be evaluated separately.
3. Rule precedence decides which competing proposals can replace earlier ones.

`precedes="generic_dynamic"` places a custom rule's evidence immediately ahead of
generic dynamic evidence. It does not put it ahead of direct metadata, special
profiles, or native-context evidence. Use `precedes="native_context"` when the rule
must supply an anchor before native inference.

`requires=("native_context",)` requires that pass to execute first, even if the
custom rule has higher evidence precedence. Built-in dependencies preserve the
established sequence of acquisition sharing, special profiles, native inference,
generic dynamics, and pre/post inference. Disabling an optional built-in pass
removes it from that sequence. A custom rule explicitly requiring a disabled pass
raises an error. Scheduling is deterministic; missing or cyclic dependencies are
rejected. Rules are evaluated once per context, not repeatedly to a fixed point.

Precedence applies to proposals actually produced by each rule. It does not bypass
the rule's applicability conditions or synthesize hypothetical alternative labels.
Phase applicability and conflicting structured DICOM phase evidence remain
protected. Timing windows and MRI profiles remain built-in policies.

## Manifest controls

The same built-in rule options work in cleaning and phase-command manifests:

```yaml
phase_curation:
  strategies:
    - type: rules
  rule_policy:
    MR:
      disabled: [native_context]
      context_columns: [SiteProtocol]
```

The named MRI rules, in default precedence order, are `metadata`,
`acquisition_phase`, `art_port_late`, `art_port`, `mask_multiart`, `native_context`,
`generic_dynamic`, and `pre_post_native`. CT has `metadata` and `pre_post_native`.
`metadata` always runs. To change built-in evidence precedence, supply a
`precedence` list containing every built-in rule for that modality exactly once,
including disabled rules. Custom rules are registered only through Python;
manifests never import or execute user code.

## Explanations and caching

`result.trace` records metadata decisions, contextual proposals, abstentions,
rejections, superseded decisions, and final canonical phases. Each entry includes
input row identity, rule name, reason, context row identities, and the selected
phase text field/value. Abstentions describe the actual execution context; the
engine does not claim a competing label from a rule that did not propose one.
The final entry identifies the canonical phase strategy. Candidate scoring
remains separate from this phase-decision trace.

`result.explain(volume_id="v12")` returns a dataframe. If volume IDs repeat, use
`result.explain(row_id=12)` with the zero-based input position instead.
`result.by_modality` exposes the legacy per-modality results, including wide
selection tables and unsupported-modality rows. Unsupported modalities have no
phase-rule trace.

The Python engine does not cache results. For caller-managed caching,
`policy.input_columns` identifies declared source dependencies and
`policy.fingerprint` identifies the configuration, rule classes, versions,
contexts, and dependencies. Bump a custom rule's version whenever its code or
settings change, and declare every external metadata column it consumes. A cache
must also fingerprint input values, row order/identity, grouping arguments, and
the installed IMPERANDI implementation. The policy fingerprint alone is not an
input-data fingerprint. CLI resume already includes normalized manifest options;
additional context columns now participate in its semantic input fingerprint.
