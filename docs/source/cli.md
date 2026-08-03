# CLI commands

The installed entry point is `imperandi`; `python -m imperandi` is equivalent.

```bash
imperandi [--log-level LEVEL] [--log-file PATH] [--quiet] COMMAND
```

Global logging options appear before the command. Project execution options
such as workers, checkpoint cadence, and resume belong in project YAML so a run
can be reproduced from its resolved configuration.

## `init`

```bash
imperandi init [PATH] [--force]
```

Creates a starter project at `PATH` (default `imperandi.yaml`). It will not
overwrite an existing file unless `--force` is supplied.

## `validate`

```bash
imperandi validate imperandi.yaml
```

Loads the built-in profile, merges project overrides, resolves paths, validates
strict typed models, and checks the stage dependency graph. On success it prints
the effective configuration SHA-256 hash.

Unknown, misspelled, and removed fields are errors. In particular,
`csv_warning_threshold_files` is an internal product heuristic and is not valid
project configuration.

## `config resolve`

```bash
imperandi config resolve imperandi.yaml
```

Prints the complete effective YAML after profile application, default filling,
and path resolution. Review or archive this output when changing a profile or
site policy.

## `plan`

```bash
imperandi plan imperandi.yaml
```

Prints the configuration hash and ordered stage graph, including required and
produced artifacts. It does not execute the pipeline.

## `run`

```bash
imperandi run imperandi.yaml
```

Executes the fixed dependency graph and prints the final `cohort_index` path.
The run directory is `<output.root>/runs/<config-hash-prefix>`.

With `execution.resume: true`, a completed stage is reused only when its
`stage.json` exists and every recorded artifact still exists. A changed
effective configuration receives a different run directory.

## `status`

```bash
imperandi status ./results/runs/<config-hash-prefix>
```

Prints every discovered stage state, including status, artifacts, and metrics.
Use `run.json` for the overall state and `stage.json` for failure detail.

## Exit behavior

Configuration, path, validation, and runtime setup errors return exit status 2
and are logged. Processing-stage exceptions mark both the current stage and the
overall run as failed before propagating to the caller.

The old step-by-step `parse`, `clean`, `ingest`, `convert`, `phase`, `segment`,
and `radiomics` commands are not part of the public CLI. Their algorithms are
orchestrated by `run`, ensuring that the same configuration and artifact
contracts govern the full cohort.
