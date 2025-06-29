import msgpack
import numpy as np
from .BaseServer import  BaseServer
from ..utils.msg import Msg, MsgType
import ray

@ray.remote
class FedAvgServer(BaseServer):

    def aggregate(self, msgs: list[Msg]) -> Msg:
        # Step 1: 类型检查
        if msgs[0].type != MsgType.GRADIENT:
            raise TypeError(f"FedAvgServer cannot process MsgType: {msgs[0].type}")


        # Step 2: 反序列化所有梯度数据
        gradients = [np.frombuffer(msg.data[0], dtype=np.float32) for msg in msgs]
        
        # Step 3: 拼接为统一数组
        updates = np.vstack(gradients)
        # Step 4: 计算平均值
        global_update = np.average(updates, axis=0)

        # Step 5: 序列化并封装成 Msg 返回
        return Msg(
            data=[global_update.tobytes()],
            type=MsgType.GRADIENT
        )

