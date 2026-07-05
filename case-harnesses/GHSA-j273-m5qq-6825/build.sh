#!/bin/sh
set -eu

classes=.nelson-harness/classes
rm -rf "$classes"
mkdir -p "$classes"

javac -encoding UTF-8 -d "$classes" \
  /harness/src/org/slf4j/Logger.java \
  /harness/src/org/slf4j/LoggerFactory.java \
  /harness/src/com/github/junrar/exception/RarException.java \
  /harness/src/com/github/junrar/rarfile/FileHeader.java \
  /harness/src/com/github/junrar/Archive.java \
  src/main/java/com/github/junrar/LocalFolderExtractor.java \
  /harness/src/com/github/junrar/BackslashTraversalWitness.java \
  /harness/src/com/github/junrar/CompatibilityControls.java
