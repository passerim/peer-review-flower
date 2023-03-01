import sys
from logging import ERROR, INFO
from typing import Callable, List, Optional

from flwr.client import Client
from flwr.common.logger import log
from flwr.server import ClientManager, History, Server, ServerConfig
from flwr.server.app import _fl, _init_defaults
from flwr.server.strategy import Strategy
from flwr.simulation.app import INVALID_ARGUMENTS_START_SIMULATION

from prflwr.simulation.transport.client_proxy import SimulationClientProxy


def start_simulation(
    *,
    client_fn: Callable[[str], Client],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
) -> History:
    """Start a sequential Flower simulation.

    Parameters
    ----------
    client_fn : Callable[[str], Client]
        A function creating client instances. The function must take a single
        str argument called `cid`. It should return a single client instance.
        Note that the created client instances are long-lived and stateful.
    num_clients : Optional[int]
        The total number of clients in this simulation. This must be set if
        `clients_ids` is not set and vice-versa.
    clients_ids : Optional[List[str]]
        List `client_id`s for each client. This is only required if
        `num_clients` is not set. Setting both `num_clients` and `clients_ids`
        with `len(clients_ids)` not equal to `num_clients` generates an error.
    server : Optional[flwr.server.Server] (default: None).
        A subclass of the class `flwr.server.Server` such as
        `prflwr.peer_review.PeerReviewServer`. If noinstance is provided,
        then `start_server` will use `flwr.server.Server`.
    config: ServerConfig (default: None).
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the abstract base class `flwr.server.ClientManager`.
        If no implementation is provided, then `start_simulation` will use
        `flwr.server.client_manager.SimpleClientManager`.

    Returns
    -------
        hist : flwr.server.history.History.
            Object containing metrics from training.
    """

    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower simulation, config: %s",
        initialized_config,
    )

    # clients_ids takes precedence
    cids: List[str]
    if clients_ids is not None:
        if (num_clients is not None) and (len(clients_ids) != num_clients):
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION)
            sys.exit()
        else:
            cids = clients_ids
    else:
        if num_clients is None:
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION)
            sys.exit()
        else:
            cids = [str(x) for x in range(num_clients)]

    # Register one SimulationClientProxy object for each client with the ClientManager
    for cid in cids:
        client_proxy = SimulationClientProxy(
            client_fn=client_fn,
            cid=cid,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    hist = _fl(
        server=initialized_server,
        config=initialized_config,
    )

    return hist
