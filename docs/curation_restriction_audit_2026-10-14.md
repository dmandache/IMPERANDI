# Curation restrictions introduced since 10 October 2026

## Scope and requested policy

This audit compares the pre-10-October curation implementation with commit
`3ddd7bf`, the October 13 MRI evidence feature branch, and the two October 14
local commits on `feat/registration-v2`. The restoration described here is
implemented on the current `feat/registration-v2` checkout. The separate
`feat/mri-dynamic-phase-evidence` branch is inspected, not modified or merged.

Requested behavior:

- Four ordered generic full-dynamic acquisitions map to NATIVE, ARTERIAL,
  PORTAL_VENOUS, DELAYED when stronger evidence does not identify their phase.
- Explicit Multiart acquisitions are arterial throughout, including bare
  Multiart descriptions without a separate T1/vendor-family keyword.
- Mask+Multiart starts native and remains arterial thereafter.
- A failed generic timing inference must not establish a permanent blocked state.
- Explicit phase labels and usable timing remain available as stronger evidence.

## Findings and disposition

| Introduced | Restriction or behavior | Consequence | Disposition in this change |
|---|---|---|---|
| 10 Oct, `3ddd7bf` | Generic unresolved inference emitted a special blocked source; later native/context passes explicitly skipped that source. | One failed heuristic could suppress otherwise available context fallback and reduce the selection score. | Removed the state, its scoring entry, and downstream veto checks. Unresolved rows retain pending status and an explanatory reason. |
| 10 Oct, `3ddd7bf` | Acquisition clocks and an assumed first arterial acquisition were used to estimate injection time. | The resulting timing windows could reject a normal ordered four-acquisition template or classify several positions as arterial based only on rapid spacing. | Removed arterial self-seeding. Without independent timing, restore the ordered phase template. |
| 10 Oct, `3ddd7bf` | Failure to fit a unique inferred timing window forced OTHER. | Timing became a gate on inference rather than optional evidence. | Use supported independent timing first; otherwise use established order. Timing reliability is false for order-only assignments. |
| 10 Oct, `3ddd7bf` | A post-only generic sequence required a compatible explicit phase anchor. | A complete ordered postcontrast sequence could remain entirely unresolved. | Removed the anchor requirement; post-only sequences use ARTERIAL, PORTAL_VENOUS, DELAYED, with later positions delayed. They do not introduce a native phase. |
| 10 Oct, `3ddd7bf` | Clock gaps above 900 seconds split a compatible group and reapplied minimum-count/anchor requirements to each fragment. | A late scan or an early native could make a supported sequence unresolved. | Removed the hard gap split from generic inference. Patient/study/date and compatibility boundaries remain. |
| 10 Oct, `3ddd7bf` | Supplied local volume order was ignored unless the split method was repeated_slice_stack. | Known caller-provided ranks could fail to establish chronology. | Restore supplied local ranks. Track provided/computed provenance so automatically generated lexical-ID ranks are still not evidence. |
| 10 Oct, `3ddd7bf` | An invalid individual acquisition clock forced OTHER even if another field established temporal order. | One malformed clock could discard a usable position in a dynamic sequence. | Degrade to order-only inference when chronology is independently available. Missing/ambiguous order remains unresolved. |
| 10 Oct, `3ddd7bf` | The broad `is_derived_low_value` flag excluded generic candidates and anchors. | MPR/derived acquisitions could lose useful phase evidence. | Generic inference excludes subtraction and quantitative/report products rather than the broad derived flag; MPR/DERIVED alone does not erase evidence. |
| 10 Oct, `3ddd7bf` | MRI diagnostic selection hard-excluded all derived-low-value candidates. | Explicitly phased MPRs could not fill otherwise empty slots. | Already corrected by `63d8fa4` and `24d4b72` on this branch. Their restoration is retained. |
| 10 Oct, `3ddd7bf` | CT selection hard-excludes derived-low-value products while retaining their phase annotations. | CT reconstructions can have a valid phase yet remain ineligible for the default selected cohort. | Audited; unchanged. The requested phase-template restoration is MRI-specific. |
| 10 Oct, `3ddd7bf` | CT localizers/bolus monitoring and MRI non-T1 sequences receive not-applicable perfusion status. | Such rows do not compete as diagnostic perfusion-phase candidates. | Retained: these are applicability boundaries, not failures to infer dynamic T1 phases. |
| 10 Oct, `3ddd7bf` | Earlier configured phase-text evidence can prevent a lower-priority field from supplying a different phase. | A broad dynamic/ordinal description can suppress a contradictory lower-priority description. | Retained as manifest-controlled evidence precedence; dynamic order fallback now works under that context. |
| 13 Oct, `7c39402`, separate MRI evidence branch | Mixed containers/ordinals require independent evidence or a validated profile; generic observed-rank progression is removed. | Four matching acquisitions, ART-PORT or Mask+Multiart can remain unresolved despite known order. | Audited as an additional tightening. This branch is not merged into the current checkout; its restrictions are not imported. |
| 13 Oct, `7c39402`, separate branch | DICOM precision/clock-domain validation rejects unsupported timing. | Less timing is accepted, but accepted clocks retain their original precision and domain. | Audited separately from phase-template policy. No precision validation is removed or copied between branches. |
| 13 Oct, `80fab42`, separate branch | First-dynamic native fallback is added after final strategy labels, with fallback confidence. | Restores native sensitivity without inventing subsequent phases. | Supportive change, not a restriction; not present on this branch and not merged as part of this task. |
| 13 Oct, `a2f4628`, separate branch | Numbered native fallback requires an actual Ph1. | A retained subset beginning with Ph2 is not relabelled native. | Audited; separate-branch behavior, not a current-branch change. |
| 14 Oct, local `6b78229` | Exact ART-PORT/ART-PORT-LATE counts were required; isolated Mask+Multiart stayed unresolved. | Additional valid acquisitions could disable all assignments. | Exact-count rule was already superseded by `4597e37`. The first and only Mask+Multiart acquisition now receives native under the requested profile assumption. |
| 14 Oct, local `4597e37` | Unsupported exact timing on a special combined profile blocks its endpoint fallback. | Special ART-PORT/ART-PORT-LATE timing conflicts remain distinguishable from missing timing. | Preserved from the earlier timing-first instruction. Generic unsupported timing now falls back to order. |

## Boundaries retained deliberately

- Explicit phase labels remain fixed and participate in chronology; generic
  positions are not renumbered after removing labelled acquisitions.
- Dixon components remain separate for temporal ranking. Known reconstructions
  of one temporal acquisition do not consume another rank.
- Patient, date, study, sequence-family and plane compatibility remain in place.
  No cross-patient or cross-study phase propagation is introduced.
- Local temporal identifiers and supplied volume ranks are not compared across
  unrelated series. `volume_order_source` distinguishes provided and computed
  ranks; lexical volume IDs and DataFrame row order are not evidence of chronology.
- The generic minimum of three distinct acquisitions, including explicit
  acquisitions, predates October 10 and is retained. Thus full dynamics with
  three positions map to native/arterial/portal; four reach delayed.
- A reliable explicit native still establishes a boundary: unlabelled
  acquisitions at or before it are not promoted into postcontrast positions.
  This does not prevent subsequent adequately ordered acquisitions resolving.
- Subtraction and quantitative/report products do not calibrate diagnostic
  timing or shift diagnostic temporal positions.
- Arterial 20–35 s, portal 60–90 s, delayed 180–900 s, and hepatobiliary policies
  originate before October 10; this change does not redefine them.

## Timing and order after restoration

1. Preserve direct labels and supported explicit delay/injection-clock decisions.
2. Establish acquisition chronology once for the compatible generic context.
3. If explicit postcontrast anchors have compatible clocks, calculate a fixed
   injection interval. An explicit native can bound injection from below; it is
   not treated as the injection timestamp.
4. Use that fixed interval when it identifies one phase. Do not calibrate it
   using an assumed arterial target or narrow it using inferred target phases.
5. Otherwise use the ordered dynamic template, retaining labelled acquisitions
   in their actual positions and preserving repeated explicit native anchors.
6. If chronology itself is absent or ambiguous, retain OTHER with pending source
   and a reason; there is no permanent veto state.

| Input context | Default result without stronger phase evidence |
|---|---|
| Four generic full-dynamic acquisitions | NATIVE → ARTERIAL → PORTAL_VENOUS → DELAYED |
| Explicit native, then three generic dynamics | NATIVE → ARTERIAL → PORTAL_VENOUS → DELAYED |
| Three explicitly post-only generic acquisitions | ARTERIAL → PORTAL_VENOUS → DELAYED |
| Multiart acquisitions | ARTERIAL throughout |
| Mask+Multiart acquisitions | First NATIVE; remaining ARTERIAL |
| ART-PORT | Existing first ARTERIAL / last PORTAL_VENOUS endpoint rule |
| ART/PORT/LATE, exactly three | ARTERIAL → PORTAL_VENOUS → DELAYED |
| Direct injection delays of 25, 32, 75, 240 seconds | ARTERIAL → ARTERIAL → PORTAL_VENOUS → DELAYED |
| No comparable chronology and no direct phase evidence | OTHER; pending with explanation |

Order-based outputs are explicitly inferred heuristics, not measured injection
delays. `mri_dynamic_timing_reliable` is true only when usable timing actually
determined the phase. Ordering metadata alone never sets that flag.

## Validation

Regression coverage exercises four-acquisition LAVA, Dixon, GRASP, TWIST and
DISCO; same-series and separate-series layouts; clocks and acquisition-order
fallback; shuffled input; Multiart and Mask+Multiart; direct injection timing;
unsupported timing fallback; long clock gaps; and unresolved chronology.
Existing tests retain coverage of explicit labels, Dixon components, temporal
positions, reconstructions, filtering, phase strategy behavior and selection.

Validation: 1,642 tests passed across `tests/unit/curation`,
`tests/unit/test_phase.py` and `tests/unit/test_clean.py`. The check covers both
CT and MRI behavior, phase strategies, exports, cleaning and prediction-command
handling. It does not measure inference accuracy on clinical image cohorts.


## Follow-up: resolve native context before dynamic progression

A separate non-dynamic native candidate is now considered before generic dynamic
inference. When no native is already resolved, the additional fallback case
requires that candidate to precede a compatible ordered dynamic sequence, with
at least three temporal acquisitions including the candidate. This establishes
native/arterial/portal with two subsequent dynamics and reaches delayed with
three. The native retains fallback confidence and is counted once. It does not
calibrate injection timing. An unrelated study, a candidate acquired after the
dynamic sequence, or other T1 volumes without dynamic evidence does not satisfy
this additional fallback case. Existing special-profile native fallback remains
available before generic inference.

Follow-up validation: 1,554 curation tests passed, including explicit/fallback
native counting, two/three following dynamics, temporal ordering, study
boundaries, unspecified same-series Dixon components, and native-only contexts.


## Reconciliation with `feat/mri-dynamic-phase-evidence`

The reconciled branch preserves the restored inference policy and selectively
ports the metadata and evidence improvements. It does not merge the restrictive
replacement of generic/special protocol ordering.

Kept: raw DICOM clock strings and subsecond/partial precision, frame/acquisition
and bolus metadata, local/UTC separation, direct DICOM phase labels and conflict
handling, acquisition-identity-based reconstruction sharing, separate GRASP
family, selected-field provenance, raw timing fingerprint inputs, and validation
of image paths only when TotalSegmentator prediction is needed. Strategies ahead
of rules contribute authoritative context before native selection and generic
progression. Actual numbered positions are retained when a known Ph1 is missing.

Not retained: manifest-configurable profiles or phase windows, mandatory profiles
or independent anchors for ordered dynamics, permanent generic blocking, a
confidence-within-10-percent selection override, or extension of hepatobiliary
timing to every delay beyond two hours. Built-in timing policies and candidate
ranking remain unchanged. Measured bolus delays must fit a unique window in full;
independent inferred-anchor timing retains the established unique-compatible
window behavior, with uncertainty propagated. Unsupported generic timing still
falls back to the established acquisition order.

Reconciliation validation on the MRI-only branch based on `main`: 1,831 tests
passed across curation, phase command, cleaning, DICOM timing/date utilities,
parsing, ingest hooks, manifests, and semantic resume. The replacement branch
does not include the separate registration feature work.


## Follow-up: native context before phase numbering

The Ph1 guard introduced during evidence reconciliation is removed. Generic
phase inference uses explicit phases/usable timing, then native context in the
compatible protocol group, then the ordered native/arterial/portal/delayed
fallback. PhN prefixes no longer impose absolute phase labels or veto native
fallback when Ph1 is absent. The secondary ordinal-to-phase resolver is removed,
so it cannot bypass the generic minimum or invent chronology for isolated PhN.

A bare contrast-agent mention and exam-level injection wording do not establish
that each dynamic volume was acquired after injection. Explicit acquisition
post-only wording (post/C+) remains stronger than native fallback. Native
candidate selection is scoped to compatible protocols and always checks that a
separate candidate precedes the dynamic/known-phase context; the legacy exam-wide
shortcut is removed. Multiple protocol groups can resolve independently. A known
water native can support unidentified reconstructions of the same protocol, but
not known fat components. Compatible explicit natives with missing clocks still
count once and prevent duplicate native fallback.

Timing now precedes the generic native assumption at every run length, removing
the separate more-than-four condition. The generic minimum of three distinct
acquisitions, special protocol mappings, explicit labels, fixed timing windows,
and absence of manifest-configurable profiles are retained.

Validation for this follow-up: 1,745 tests passed across curation, the phase
command, cleaning, and DICOM timing. Coverage includes missing/mixed phase
numbers, contrast-agent wording, explicit/fallback/clockless native context,
unrelated and multiple protocol groups, chronological candidate eligibility,
and supported timing on plain T1 acquisitions.
