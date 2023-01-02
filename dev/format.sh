#!/bin/bash
echo "=== format.sh ==="
python -m isort examples prflwr tests               && echo "- isort:         done" &&
python -m black -q examples prflwr tests            && echo "- black:         done" &&
python -m docformatter -i -r examples prflwr tests  && echo "- docformatter:  done" &&
echo "- Code formatted"
