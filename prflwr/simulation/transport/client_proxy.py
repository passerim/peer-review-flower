from logging import DEBUG
from typing import Optional, cast

from flwr import common
from flwr.client import Client, to_client
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.ray_transport.ray_client_proxy import ClientFn


class SimulationClientProxy(ClientProxy):
    def __init__(self, client_fn: ClientFn, cid: str):
        super().__init__(cid)
        self.client: Client = to_client(client_fn(cid))

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        try:
            res = self.client.get_properties(ins)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        return cast(
            common.GetPropertiesRes,
            res,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        try:
            res = self.client.get_parameters(ins)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        return cast(
            common.GetParametersRes,
            res,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        try:
            res = self.client.fit(ins)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        return cast(
            common.FitRes,
            res,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        try:
            res = self.client.evaluate(ins)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        return common.DisconnectRes(reason="")
