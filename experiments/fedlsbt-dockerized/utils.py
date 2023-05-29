import json
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import Tree


def tree_to_state_dict(tree: Tree) -> Dict[str, Any]:
    serialized_tree = tree.__getstate__()
    dtypes = serialized_tree["nodes"].dtype
    serialized_tree["nodes_dtype"] = [dtypes[i].str for i in range(0, len(dtypes))]
    serialized_tree["nodes"] = serialized_tree["nodes"].tolist()
    serialized_tree["values"] = serialized_tree["values"].tolist()
    return serialized_tree


def decision_tree_regressor_to_state_dict(
    model: DecisionTreeRegressor,
) -> Dict[str, Any]:
    serialized_model = model.__getstate__()
    tree = tree_to_state_dict(model.tree_)
    serialized_model["feature_names_in_"] = model.feature_names_in_.tolist()
    serialized_model["tree_"] = tree
    return serialized_model


def serialize_decision_tree_regressor(model: DecisionTreeRegressor) -> bytes:
    model_state_dict = decision_tree_regressor_to_state_dict(model)
    json_formatted = json.dumps(model_state_dict)
    return bytes(json_formatted, "utf-8")


def state_dict_to_tree(
    tree_dict: Dict[str, Any], n_features: int, n_classes: int, n_outputs: int
):
    tree_dict["nodes"] = [tuple(lst) for lst in tree_dict["nodes"]]
    names = [
        "left_child",
        "right_child",
        "feature",
        "threshold",
        "impurity",
        "n_node_samples",
        "weighted_n_node_samples",
    ]
    tree_dict["nodes"] = np.array(
        tree_dict["nodes"],
        dtype=np.dtype({"names": names, "formats": tree_dict["nodes_dtype"]}),
    )
    tree_dict["values"] = np.array(tree_dict["values"])
    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)
    return tree


def state_dict_to_decision_tree_regressor(model_dict: Dict[str, Any]):
    deserialized_decision_tree = DecisionTreeRegressor()
    for key, value in iter(model_dict.items()):
        if key == "feature_names_in_":
            setattr(deserialized_decision_tree, key, np.asarray(value))
        elif key != "tree_":
            setattr(deserialized_decision_tree, key, value)
    tree = state_dict_to_tree(
        model_dict["tree_"], model_dict["n_features_in_"], 1, model_dict["n_outputs_"]
    )
    deserialized_decision_tree.tree_ = tree
    return deserialized_decision_tree


def deserialize_decision_tree_regressor(
    model_serialized: bytes,
) -> DecisionTreeRegressor:
    json_formatted = str(model_serialized, "utf-8")
    model_state_dict = json.loads(json_formatted)
    return state_dict_to_decision_tree_regressor(model_state_dict)


class FederatedGradientBoostingRegressor:
    def __init__(self):
        self.estimators_ = []
        self.gammas_ = []

    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        for estimator, gamma in zip(self.estimators_, self.gammas_):
            y_pred = y_pred + gamma * estimator.predict(x)
        return y_pred

    def add_estimator(self, estimator, gamma):
        if not self.estimators_:
            self.max_features_ = estimator.max_features_
            self.n_features_in_ = estimator.n_features_in_
        for attr in ["max_features_", "n_features_in_"]:
            if getattr(self, attr) != getattr(estimator, attr):
                return
        self.estimators_.append(estimator)
        self.gammas_.append(gamma)


def gradient_boosting_to_state_dict(
    model: FederatedGradientBoostingRegressor,
) -> Dict[str, Any]:
    serialized_model = {
        "estimators_": [],
        "gammas_": model.gammas_,
    }
    if getattr(model, "max_features_", None):
        serialized_model["max_features_"] = model.max_features_
    if getattr(model, "n_features_in_", None):
        serialized_model["n_features_in_"] = model.n_features_in_
    for tree in model.estimators_:
        serialized_model["estimators_"].append(
            decision_tree_regressor_to_state_dict(tree)
        )
    return serialized_model


def state_dict_to_gradient_boosting(
    model_dict: Dict[str, Any]
) -> FederatedGradientBoostingRegressor:
    model = FederatedGradientBoostingRegressor()
    model.estimators_ = [
        state_dict_to_decision_tree_regressor(tree)
        for tree in model_dict["estimators_"]
    ]
    model.gammas_ = model_dict["gammas_"]
    if model_dict.get("max_features_", False):
        model.max_features_ = model_dict["max_features_"]
    if model_dict.get("max_features_", False):
        model.n_features_in_ = model_dict["n_features_in_"]
    return model


def serialize_gradient_boosting_regressor(
    model: FederatedGradientBoostingRegressor,
) -> bytes:
    json_formatted = json.dumps(gradient_boosting_to_state_dict(model))
    return bytes(json_formatted, "utf-8")


def deserialize_gradient_boosting_regressor(
    model_serialized: bytes,
) -> FederatedGradientBoostingRegressor:
    json_formatted = str(model_serialized, "utf-8")
    model_dict = json.loads(json_formatted)
    return state_dict_to_gradient_boosting(model_dict)


def serialize_global_model_and_candidates(
    model: Union[FederatedGradientBoostingRegressor, bytes],
    candidates: Union[List[DecisionTreeRegressor], bytes],
) -> bytes:
    if isinstance(model, bytes):
        model = deserialize_gradient_boosting_regressor(model)
    if isinstance(candidates[0], bytes):
        candidates = [deserialize_decision_tree_regressor(tree) for tree in candidates]
    serialized_models = dict(
        global_model=gradient_boosting_to_state_dict(model),
        candidate_models=[
            decision_tree_regressor_to_state_dict(tree) for tree in candidates
        ],
    )
    json_serialized = json.dumps(serialized_models)
    return bytes(json_serialized, "utf-8")


def load_data(
    cid: int,
    num_clients: int,
    data_home: str = "./data",
):
    data_x, data_y = fetch_california_housing(
        data_home=data_home, return_X_y=True, as_frame=True
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, shuffle=True, random_state=0
    )
    x_train, y_train = x_train[(y_train < 5)], y_train[(y_train < 5)]
    x_test, y_test = x_test[(y_test < 5)], y_test[(y_test < 5)]
    if cid > -1:
        kmeans = KMeans(num_clients)
        train_clusters = kmeans.fit_predict(
            x_train[["Latitude", "Longitude"]].to_numpy()
        )
        test_clusters = kmeans.predict(x_test[["Latitude", "Longitude"]].to_numpy())
        return (
            x_train[train_clusters == cid],
            x_test[test_clusters == cid],
            y_train[train_clusters == cid],
            y_test[test_clusters == cid],
        )
    else:
        return x_train, x_test, y_train, y_test
