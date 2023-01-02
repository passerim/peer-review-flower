#!/bin/bash
echo "=== test.sh ==="
python -m isort --check-only examples prflwr tests  && echo "- isort:         done" &&
python -m black -q --check examples prflwr tests    && echo "- black:         done" &&
python -m docformatter -c -r examples prflwr tests  && echo "- docformatter:  done" &&
python -m flake8 .                                  && echo "- flake8:        done" &&
timeout 300 python -m unittest                      && echo "- unittest:      done" &&
echo "- All Python checks passed"
