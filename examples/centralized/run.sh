#!/bin/bash

if [ -f centralized.py ]; then cd ../..; fi

# Executed cifar-10 centralized training script
python -m examples.centralized.centralized &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
