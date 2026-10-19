"""Context partitioning shared by CT and MRI custom rules."""

from dataclasses import replace

from .metadata_helpers import stable_text


def custom_contexts(context, scope, *, protocol_key=None):
    rows = context.rows
    if scope == "row":
        groups = [[idx] for idx in rows.index]
    else:
        # Selection groups can intentionally span exams. Extension rules must
        # still respect all available patient/study boundaries.
        boundary = [
            c
            for c in ("patient_key", "date", "study_id", "StudyInstanceUID")
            if c in rows
        ]
        keys = [rows[c].map(stable_text) for c in boundary]
        if scope == "compatible_protocol":
            if protocol_key is not None:
                keys.append(rows.apply(protocol_key, axis=1))
            else:
                keys.extend(
                    rows[c].map(stable_text)
                    for c in ("ProtocolName", "acquisition_plane")
                    if c in rows
                )
        groups = (
            rows.groupby(keys, dropna=False, sort=False).groups.values()
            if keys
            else [rows.index]
        )
    for indices in groups:
        yield replace(
            context,
            rows=rows.loc[indices].copy(),
            scope=scope,
            authoritative_rows=context.authoritative_rows.intersection(indices),
        )
