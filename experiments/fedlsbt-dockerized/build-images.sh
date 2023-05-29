#!/bin/bash

# Build client image
docker build . --no-cache --file ./images/client/Dockerfile --tag fedlsbt-client:latest
echo "- Client image built"

# Build server image
docker build . --no-cache --file ./images/server/Dockerfile --tag fedlsbt-server:latest
echo "- Server image built"
