import msgpack
import tenseal as ts
from BaseServer import  BaseServer
from src.ParaAegis.utils.msg import Msg, MsgType
import ray

@ray.remote
class HeAvgServer(BaseServer):

    def __init__(self, ctx_bytes: bytes):
        self.he_ctx = ts.context_from(ctx_bytes)

    def aggregate(self, msgs: list[Msg]) -> Msg:
        # Step 1: 类型检查
        if msgs[0].type != MsgType.ENCRYPTED_GRADIENT:
            raise TypeError(f"FedAvgServer cannot process MsgType: {msgs[0].type}")

        # Step 2: 反序列化所有梯度数据
        gradients = [ts.ckks_vector_from(self.he_ctx, msg.data[0]) for msg in msgs]

        aggregated = None
        for grad in gradients:
            if aggregated is None:
                aggregated = grad
            else:
                aggregated += grad

        # Step 5: 序列化并封装成 Msg 返回
        return Msg(
            data=[aggregated.serialize()],
            type=MsgType.ENCRYPTED_GRADIENT
        )

