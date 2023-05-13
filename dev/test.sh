#!/bin/bash

# Stop unittest execution after MAX_TIME and return an error.
if [[ -z "$1" ]]; then
    MAX_TIME=300
else
    MAX_TIME="$1"
fi

# Command 'set -e' makes shell exit immediately if a command exits with a non-null status.
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="
python -m isort --check-only examples prflwr tests  && echo "- isort:         done" &&
python -m black -q --check examples prflwr tests    && echo "- black:         done" &&
python -m docformatter -c -r examples prflwr tests  && echo "- docformatter:  done" &&
python -m flake8 .                                  && echo "- flake8:        done" &&
timeout $MAX_TIME python -m unittest                && echo "- unittest:      done" &&
echo "- All Python checks passed"
