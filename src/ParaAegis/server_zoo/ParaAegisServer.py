import msgpack
import numpy as np
from .BaseServer import  BaseServer
from ..utils.msg import Msg, MsgType
import ray
import tenseal as ts

@ray.remote
class ParaAegisServer(BaseServer):
    def __init__(self, grad_len: int, he_frac, he_ctx_bytes: bytes) -> None:
        super().__init__()
        self.grad_len = grad_len
        self.he_frac = he_frac  # Fraction of HE data to use
        self.he_ctx = ts.context_from(he_ctx_bytes)  # Initialize the HE context from the provided bytes

    def aggregate(self, msgs: list[Msg]) -> Msg:
        
        if msgs is None or len(msgs) == 0:
            raise ValueError("No messages to aggregate")
        
        if not all(isinstance(msg, Msg) for msg in msgs):
            raise TypeError("All items in msgs must be of type Msg")
    

        # Step 1: 类型检查
        if msgs[0].type == MsgType.VOTE:
            return self._aggregate_vote(msgs)
        elif msgs[0].type == MsgType.PARTIAL_ENCRYPTED_GRADIENT:
            return self._aggregate_grad(msgs)
        else:
            raise TypeError(f"ParaAegisServer cannot process MsgType: {msgs[0].type}")

    def _aggregate_grad(self, msgs: list[Msg]) -> Msg:
        """
        聚合梯度消息，计算平均梯度并返回新的 Msg 对象。
        :param msgs: 包含梯度数据的 Msg 对象列表
        :return: 包含平均梯度数据的 Msg 对象
        """
        he_parts = [ts.ckks_vector_from(self.he_ctx, msg.data[0]) for msg in msgs]
        dp_parts = np.vstack([np.frombuffer(msg.data[1], dtype=np.float32) for msg in msgs])
        
        he_global = None
        for he_part in he_parts:
            if he_global is None:
                he_global = he_part
            else:
                he_global += he_part
        he_global *= (1 / len(he_parts))  # 平均化 HE 部分
        
        dp_global = np.mean(dp_parts, axis=0).astype(np.float32)
        
        he_bytes = he_global.serialize()  # 序列化 HE 部分
        dp_bytes = dp_global.tobytes()  # 序列化 DP 部分

        return Msg(
            data=[he_bytes, dp_bytes],
            type=MsgType.PARTIAL_ENCRYPTED_GRADIENT
        )
        
        
    def _aggregate_vote(self, msgs: list[Msg]) -> Msg:
        """
        聚合投票消息，计算最终投票结果并返回新的 Msg 对象。
        :param msgs: 包含投票数据的 Msg 对象列表
        :return: 包含最终投票结果的 Msg 对象
        """
        # Step 1: 类型检查
        if not all(isinstance(msg, Msg) for msg in msgs):
            raise TypeError("All items in msgs must be of type Msg")
        
        votes = np.zeros(self.grad_len, dtype=np.int32)

        for msg in msgs:
            vote = np.frombuffer(msg.data[0], dtype=np.int32)
            votes[vote] += 1
        
        he_count = int(self.grad_len * self.he_frac)
        # Step 3: 计算最终投票结果
        final_votes = np.argpartition(votes, -he_count)[-he_count:].astype(np.int32)

        # Step 4: 封装成 Msg 返回
        return Msg(
            data=[final_votes.tobytes()],
            type=MsgType.VOTE
        )
