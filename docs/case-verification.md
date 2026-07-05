# Executable case verification

Nelson can validate a case-specific security harness against both revisions of a
known vulnerability. A valid harness has to demonstrate all of these properties:

- every build step succeeds on the vulnerable and fixed commits;
- every security witness has the declared red outcome on the vulnerable commit;
- the same witnesses have the declared green outcome on the upstream fix; and
- compatibility controls remain green on both commits.

## Terminology

**Security harness:** The trusted, case-specific collection of build steps,
security witnesses, compatibility controls, and supporting fixtures used to test
a vulnerability. It is benchmark metadata kept hidden from the model being
evaluated.

**Security witness:** An executable check that demonstrates a specific security
invariant. Its declared result differs between revisions: it exposes the unsafe
behavior on the vulnerable revision and passes when that behavior is prevented by
the fixed revision or a valid candidate mitigation.

Harness commands run as argument arrays in rootless Podman containers, never
through a shell. Source checkouts are separate and writable because build systems
need to create artifacts. Container networking is disabled by default.

## Case metadata

Add a `verification` mapping to a case manifest:

```yaml
verification:
  invariant: Untrusted paths cannot escape the extraction root
  # Optional; defaults to Nelson's benchmark image.
  image: nelson-bench:fedora-tools2
  # Enable only when the build genuinely requires it. Prefer prebuilt images.
  network: false
  build:
    - argv: [make, all]
      timeout_s: 600
  witnesses:
    - name: traversal-primary
      command:
        argv: [python3, /harness/test_traversal.py]
        timeout_s: 60
      vulnerable_exit_codes: [1]
      fixed_exit_codes: [0]
    - name: traversal-encoded-variant
      command: [python3, /harness/test_encoded_traversal.py]
      vulnerable_exit_codes: [1]
      fixed_exit_codes: [0]
  controls:
    - argv: [python3, /harness/test_safe_extract.py]
```

Keep authoritative harness files outside the source checkout and pass their
directory with `--harness-dir`; Nelson mounts it read-only at `/harness` in both
containers. Upstream regression tests are useful witness seeds, but copy and
adapt them into this independent harness, then add a witness variant and a
legitimate-behavior control before treating the case as strong proof.

## Running verification

With a case stored in the database:

```bash
nelson corpus verify GHSA-j273-m5qq-6825 \
  --harness-dir case-harnesses/GHSA-j273-m5qq-6825
```

Or resolve a case directly from manifests:

```bash
nelson corpus verify GHSA-j273-m5qq-6825 --from-manifest cases \
  --harness-dir case-harnesses/GHSA-j273-m5qq-6825
```

The command exits nonzero for malformed metadata, checkout/build infrastructure
errors, unexpected witness behavior, or failing controls. It does not modify the
case's vetted status; executable verification is an additional evidence tier.

## Verifying a candidate mitigation

After certifying the harness against the known vulnerable and fixed revisions,
apply a model-generated unified diff to a fresh vulnerable checkout and run the
fixed expectations:

```bash
nelson corpus verify-patch GHSA-j273-m5qq-6825 \
  --patch candidate.diff \
  --from-manifest cases \
  --harness-dir case-harnesses/GHSA-j273-m5qq-6825
```

Nelson reports patch application, build, security witnesses, and compatibility
controls separately. A patch that does not apply is a candidate failure, while a
checkout or container failure is an infrastructure error. The cached vulnerable
checkout remains pristine: every invocation copies it into a new candidate tree
before applying the patch.

Candidate patches are limited to 5 MiB and must be plain unified diffs accepted by
`git apply`; Markdown fences and prose are intentionally rejected. Passing this
command proves the mitigation satisfies the case harness. It does not by itself
prove that the model's written vulnerability report was accurate or complete.

## Finding-scoped remediation jobs

Detection and remediation are separate benchmark jobs. Given one finding produced
by a completed detection run, Nelson can start a fresh model job to inspect the
vulnerable repository, generate a patch, and immediately run the hidden harness:

```bash
nelson bench remediate FINDING_ID \
  --competitor raw-api-loop/gemma4-31b \
  --harness-dir case-harnesses/GHSA-j273-m5qq-6825 \
  --thinking \
  --max-output-tokens 16384
```

The remediation model receives that finding's location and explanation, not the
authoritative case description, upstream fix, or hidden harness. Each attempt is
stored independently in `remediation_runs`, including its configuration, token and
time usage, raw response, extracted patch, and separate patch/build/witness/control
outcomes. A retry therefore adds evidence instead of replacing a failed attempt.
