#!/bin/bash

set -e
trap 'echo "Command failed: $BASH_COMMAND"' ERR

# Make the current directory the directory of this script.
pushd "$(dirname "$(readlink -f -- "${BASH_SOURCE[0]}" || realpath -- "${BASH_SOURCE[0]}")")"
# On exit, change the current directory back to the original directory.
trap popd EXIT

rm -rf _build
mkdir _build
cp -dR source _build/
cp -dR examples/* _build/source/examples/
sphinx-apidoc -e -o _build/source/_modules ../simpml
sphinx-build -M html _build/source _build
