# Peer-reviewed Federated Learning in Flower

Peer reviewed federated learning usign Flower.

## Installation

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. 

We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)).

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Class and sequence diagrams

### Sequence diagram of the federated training step:

![Sequence diagram](/imgs/sequenza.png)

### Class diagram of the federated training step:

![Class diagram](/imgs/classi.png)

## Examples

To run the centralized, federated and federated with peer review training examples, run the ```run.sh``` scripts in their respective directories.

## Tests

To run the centralized test:
```shell
python -m tests.test_centralized
```

To run the federated test:
```shell
python -m tests.test_federated
```

To run the peer reviewed federated tests:
```shell
python -m tests.test_peer_reviewed
python -m tests.test_fed_pr_equivalence
```
