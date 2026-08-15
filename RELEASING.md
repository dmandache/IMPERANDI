# Releasing IMPERANDI

Releases are built from Git tags. The package version is derived from the tag,
so `pyproject.toml` does not need a manual version edit.

## One-time PyPI setup

1. Create a GitHub environment named `pypi` and require a reviewer before
   deployment.
2. In PyPI's Trusted Publisher settings, register this GitHub Actions publisher:

   - PyPI project: `imperandi`
   - Owner: `dmandache`
   - Repository: `IMPERANDI`
   - Workflow: `release.yml`
   - Environment: `pypi`

No PyPI API token or repository secret is required.

## Publish a release

Start from a commit on `main` whose normal test workflow has passed. Create and
push an annotated PEP 440 version tag with a `v` prefix:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

The release workflow then:

1. runs lint and unit tests on Python 3.10, 3.11, and 3.12;
2. builds and validates the wheel and source distribution;
3. confirms the built package version matches the tag;
4. waits for approval in the `pypi` environment and publishes with Trusted
   Publishing; and
5. creates a GitHub Release with generated notes and both distributions
   attached.

PyPI release files are immutable. Do not move or reuse a published tag; create a
new patch release instead.
