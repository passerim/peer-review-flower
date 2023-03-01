#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== build-docs.sh ==="
cd doc
make clean
make html
echo "- Documentation built under doc/build/"
