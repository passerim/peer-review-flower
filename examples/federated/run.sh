#!/bin/bash

PORT=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3

cd ../../

python -m src.federated.server --port $PORT --num_rounds $NUM_ROUNDS &

# Sleep for 5s to give the server enough time to start
sleep 5

for (( i=0; i<=$NUM_CLIENTS-1; i++ ))
do
    echo "Starting client $i"
    python -m src.federated.client --port $PORT --num_clients $NUM_CLIENTS --partition $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
