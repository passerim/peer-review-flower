#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== format.sh ==="
python -m isort examples prflwr tests               && echo "- isort:         done" &&
python -m black examples prflwr tests               && echo "- black:         done" &&
python -m docformatter -i -r examples prflwr tests  && echo "- docformatter:  done" &&
echo "- Code formatted"
