# Executable case verification

Nelson can validate a case-specific security harness against both revisions of a
known vulnerability. A valid harness has to demonstrate all of these properties:

- every build step succeeds on the vulnerable and fixed commits;
- every security witness has the declared red outcome on the vulnerable commit;
- the same witnesses have the declared green outcome on the upstream fix; and
- compatibility controls remain green on both commits.

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
