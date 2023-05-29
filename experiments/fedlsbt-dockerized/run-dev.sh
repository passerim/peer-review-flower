#!/bin/bash

# Load environment variables from .env file
set -a; source .env; set +a
SERVER_ADDRESS=localhost:$SERVER_PORT

# Start scripts from parent package
if [ -f client.py ]; then cd ../..; fi

# Set development env variable
export RUN_DEV=True

echo "Starting server"
python -m experiments.fedlsbt-dockerized.server &

# Sleep for 5s to give the server enough time to start
sleep 5

for (( i=0; i<=$NUM_CLIENTS-1; i++ ))
do
    echo "Starting client $i"
    python -m experiments.fedlsbt-dockerized.client --cid $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
