import argparse
import json
import os
from copy import deepcopy as cp
from typing import Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
)
from sklearn.metrics import mean_squared_error
from sklearn.tree import ExtraTreeRegressor

from prflwr.peer_review import (
    PeerReviewClient,
    ReviewIns,
    ReviewRes,
    TrainIns,
    TrainRes,
)

# Import utility functions
if os.getenv("RUN_DEV"):
    from .utils import *
else:
    from utils import *

# Useful constants
OK_STATUS = Status(Code.OK, "")
DEFAULT_MAX_LEAVES = 8


class FedLSBTClient(PeerReviewClient):
    def __init__(
        self,
        cid: int,
        trainset: Tuple[np.ndarray, np.array],
        testset: Tuple[np.ndarray, np.array],
    ):
        self.cid = cid
        self.trainset = trainset
        self.testset = testset

    def train(self, ins: TrainIns) -> TrainRes:
        parameters, config = ins.parameters, ins.config
        model = deserialize_gradient_boosting_regressor(parameters.tensors[0])
        ri = self.trainset[1] - model.predict(self.trainset[0])
        tree = ExtraTreeRegressor(
            max_leaf_nodes=config.get("max_leaves", DEFAULT_MAX_LEAVES),
            random_state=self.cid**2,
        )
        tree.fit(self.trainset[0], ri)
        res = TrainRes(
            OK_STATUS,
            Parameters(
                tensors=[serialize_decision_tree_regressor(tree)], tensor_type="bytes"
            ),
            len(self.trainset[1]),
            {},
        )
        return res

    def review(self, ins: ReviewIns) -> ReviewRes:
        parameters = ins.parameters
        state_dicts = json.loads(str(parameters.tensors[0], "utf-8"))
        model = state_dict_to_gradient_boosting(state_dicts["global_model"])
        candidates = [
            state_dict_to_decision_tree_regressor(tree)
            for tree in state_dicts["candidate_models"]
        ]
        res_model = self.trainset[1] - model.predict(self.trainset[0])
        res_candidates = [tree.predict(self.trainset[0]) for tree in candidates]
        res_candidates = np.stack(res_candidates, axis=1).T
        numerator_k = res_candidates @ res_model
        denominator_k = res_candidates @ res_candidates.T
        res = ReviewRes(
            OK_STATUS,
            ndarrays_to_parameters([numerator_k, denominator_k]),
            len(self.trainset[1]),
            {},
        )
        return res

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters, _ = ins.parameters, ins.config
        model = deserialize_gradient_boosting_regressor(parameters.tensors[0])
        y_pred = model.predict(self.testset[0])
        loss = mean_squared_error(self.testset[1], y_pred)
        return EvaluateRes(OK_STATUS, loss, len(self.testset[1]), {})


def main(args):
    cid = int(args.cid)
    x_train, x_test, y_train, y_test = load_data(cid, args.num_clients)
    trainset = (x_train, y_train)
    testset = (x_test, y_test)
    client = FedLSBTClient(cid, cp(trainset), cp(testset))
    fl.client.start_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cid",
        type=int,
        default=os.environ.get("CID", None),
        required=False,
        help="client id of the current client",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=os.environ.get("NUM_CLIENTS", None),
        required=False,
        help="number of clients participating to the training",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default=os.environ.get("SERVER_ADDRESS", "localhost:8080"),
        required=False,
        help="ip address of the server formatted as: <host:port>",
    )
    args = parser.parse_args()
    if args.cid is None:
        raise ValueError(
            "It is necessary to specify the client id of the current client."
        )
    elif args.num_clients is None:
        raise ValueError(
            "It is necessary to specify the number of clients participating to the training."
        )
    main(args)
