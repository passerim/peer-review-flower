#!/bin/bash

# Load environment variables from .env file
set -a; source .env; set +a
SERVER_ADDRESS=$SERVER_HOST:$SERVER_PORT

# Build images if not already built
if [ $(docker images fedlsbt-client | wc -l) -eq 1 ]; then
    ./build-images.sh
elif [ $(docker images fedlsbt-server | wc -l) -eq 1 ]; then
    ./build-images.sh
fi

echo "- Starting server"
docker run -d --rm --env-file .env --env SERVER_ADDRESS=$SERVER_ADDRESS --ip $SERVER_HOST --name fedlsbt-server fedlsbt-server

# Sleep for 5s to give the server enough time to start
sleep 5

for (( i=0; i<=$NUM_CLIENTS-1; i++ ))
do
    echo "- Starting client $i"
    docker run -d --rm --env-file .env --env CID=$i --env SERVER_ADDRESS=$SERVER_ADDRESS fedlsbt-client
done

# This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Attach to server running container
docker attach fedlsbt-server
