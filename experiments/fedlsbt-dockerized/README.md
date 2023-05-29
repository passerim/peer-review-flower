# Distributed FedLSBT

This experiment is about an example of a distributed federated learning setup that features some clients, a server and uses the FedLSBT algorithm to train a federated GBDT regression model.

In order to minimize the amount of setup and potential issues that might arise due to the hardware/software heterogenity between clients, we'll be running the clients inside a Docker. Even the server is dokerized to allow for rapid experimenting and setup customization on the fly.

## Running the experiment

To run the experiment it is necessary to have Docker installed, then you can to start the experiment running:
```
./run-docker.sh
```

It is possible to customize the federated learning setup by modifying the environment variables inside the ```.env``` file.
