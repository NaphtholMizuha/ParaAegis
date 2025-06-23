import msgpack
import tenseal as ts
from tenseal import CKKSVector

from .BaseServer import  BaseServer
from ..utils import suppress_stdout
from ..utils.msg import Msg, MsgType
import ray

@ray.remote
class HeAvgServer(BaseServer):

    def __init__(self, ckks_bytes: bytes):
        self.he_ctx = ts.context_from(ckks_bytes)

    def aggregate(self, msgs: list[Msg]) -> Msg:
        # Step 1: 类型检查
        if msgs[0].type != MsgType.ENCRYPTED_GRADIENT:
            raise TypeError(f"FedAvgServer cannot process MsgType: {msgs[0].type}")
        length = len(msgs)
        # Step 2: 反序列化所有梯度数据
        with suppress_stdout():
            gradients = [ts.ckks_vector_from(self.he_ctx, msg.data[0]) for msg in msgs]

        aggregated = None
        for grad in gradients:
            if aggregated is None:
                aggregated = grad
            else:
                aggregated += grad
        aggregated *= (1 / length)
        # Step 5: 序列化并封装成 Msg 返回
        return Msg(
            data=[aggregated.serialize()],
            type=MsgType.ENCRYPTED_GRADIENT
        )

