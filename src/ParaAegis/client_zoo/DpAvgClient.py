import ray

from .BaseClient import BaseClient
from ..utils import Msg, MsgType
from ..training.trainer import Trainer
import numpy as np
import tenseal as ts
import logging
logging.getLogger("tenseal").setLevel(logging.ERROR)
@ray.remote(num_gpus=0.1)
class DpAvgClient(BaseClient):
    def __init__(
        self,
        trainer: Trainer,
        n_epochs: int,
        noise_multiplier: float = 1.0,
        clip_thr: float = 1.0
    ):
        """
        Initializes the DpAvgClient with input and output message types, a trainer object, the number of training epochs,
        and a noise multiplier for differential privacy.

        :param trainer: The trainer object responsible for training the model.
        :param n_epochs: The number of epochs to train the model locally.
        :param noise_multiplier: The multiplier for the noise added to gradients for differential privacy.
        """
        self.trainer = trainer
        self.n_epochs = n_epochs
        self.noise_multiplier = noise_multiplier
        self.clip_thr = clip_thr
        
    def get(self, msg_type: MsgType) -> Msg:
        grad = self.trainer.get_grad()  # Get the gradient from the trainer
        idcs = np.argsort(-np.abs(grad))  # Get indices of the gradient components sorted by absolute value
        temp = grad.copy()
        i = 0

        while i < len(grad) and np.linalg.norm(temp) > self.clip_thr:
            temp[idcs[i]] = 0  # Set the i-th largest gradient component to zero
            i += 1

        idcs = idcs[i:]  # Keep only the indices of the smallest components

        grad[idcs] += np.random.normal(0, self.noise_multiplier, size=idcs.shape)  # Add Gaussian noise for differential privacy
        grad_bytes = grad.tobytes()
        if not isinstance(grad_bytes, bytes):
            raise TypeError("Serialized gradient must be of type 'bytes'")
        return Msg([grad_bytes], MsgType.GRADIENT)  # type

    def set(self, in_msg: Msg):
        if in_msg.type == MsgType.GRADIENT:
            grad = np.frombuffer(in_msg.data[0], dtype=np.float32)  # Deserialize the gradient from the message
            if not isinstance(grad, np.ndarray):
                raise TypeError("Deserialized gradient must be of type 'dict'")
            self.trainer.set_grad(grad)  # Set the gradient back to the trainer
        else:
            raise TypeError(f"FedAvgClient cannot process type: {in_msg.type}")  # Raise an error for unsupported message types

    def run(self):
        """
        Performs local training on the model for the specified number of epochs.

        This method calls the local_train method of the trainer object, passing the number of epochs.
        """
        self.trainer.local_train(self.n_epochs)  # Perform local training for the specified number of epochs
        self.grad = self.trainer.get_grad()  # Retrieve the gradient after training

    def test(self):
        return self.trainer.test()