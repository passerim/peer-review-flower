#!/bin/bash

if [ -f simulation.py ]; then cd ../..; fi

python -m examples.sim_peer_reviewed.simulation $@ &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
