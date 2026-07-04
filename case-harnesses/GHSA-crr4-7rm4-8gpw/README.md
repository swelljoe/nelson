# GHSA-crr4-7rm4-8gpw verification harness

This harness executes Mastodon's real `PrivateAddressCheck` module. The witness
asserts that the IPv6 unspecified address `::` is rejected as private before it
can reach an outbound connection. The vulnerable commit misclassifies it; the
upstream fixed commit includes `::/128` in the prohibited CIDR list.

The controls ensure the patch preserves classification of ordinary public IPv4
and IPv6 addresses while continuing to block loopback, private, and link-local
addresses.

Build the pinned execution image:

```bash
podman build -t nelson-verify-mastodon:ruby-3.3 \
  -f case-harnesses/GHSA-crr4-7rm4-8gpw/Containerfile \
  case-harnesses/GHSA-crr4-7rm4-8gpw
```

Run the differential proof without importing the manifest into the main DB:

```bash
nelson corpus verify GHSA-crr4-7rm4-8gpw \
  --db /tmp/nelson-mastodon-verify.db \
  --from-manifest cases \
  --harness-dir case-harnesses/GHSA-crr4-7rm4-8gpw
```

Expected result: the witness exits 1 on the vulnerable revision and 0 on the
fixed revision; the compatibility controls exit 0 on both.
