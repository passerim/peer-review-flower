#!/bin/bash

PORT=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3

if [ -f client.py ]; then cd ../..; fi

python -m examples.peer_reviewed.server --port $PORT --num_rounds $NUM_ROUNDS &

# Sleep for 5s to give the server enough time to start
sleep 5

for (( i=0; i<=$NUM_CLIENTS-1; i++ ))
do
    echo "Starting client $i"
    python -m examples.peer_reviewed.client --port $PORT --cid $i --num_clients $NUM_CLIENTS &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
