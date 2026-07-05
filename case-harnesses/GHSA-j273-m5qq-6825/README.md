# Junrar backslash traversal verification

This harness compiles the repository's real `LocalFolderExtractor.java` with minimal
stubs for unrelated Junrar and SLF4J APIs. It does not use the upstream regression
test or its archive fixture.

The witness supplies `..\\escaped.txt`, then checks the filesystem effect. On the
vulnerable revision the validation treats the backslash as a literal character,
while file creation later treats it as a separator and writes outside the extraction
directory. The fixed revision normalizes separators before validation and rejects it.

The controls require ordinary and nested extraction to keep working and confirm that
the already-protected forward-slash traversal remains rejected on both revisions.

Build the pinned verifier image and run the differential check:

```sh
podman build -t nelson-verify-junrar:jdk-17 \
  -f case-harnesses/GHSA-j273-m5qq-6825/Containerfile \
  case-harnesses/GHSA-j273-m5qq-6825

nelson corpus verify GHSA-j273-m5qq-6825 \
  --from-manifest cases \
  --harness-dir case-harnesses/GHSA-j273-m5qq-6825
```
