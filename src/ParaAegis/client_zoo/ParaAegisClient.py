from BaseClient import BaseClient
from ..utils.msg import MsgType, Msg
from src.ParaAegis.training.trainer import Trainer
import ray
import tenseal as ts

@ray.remote()
class ParaAegisClient(BaseClient):

    def __init__(
        self,
        trainer: Trainer,
        n_epochs: int,
        he_config: dict,
    ):

        self.trainer = trainer  # Initialize the trainer object
        self.n_epochs = n_epochs  # Set the number of training epochs
        self.he_ctx = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=he_config['poly'],
            coeff_mod_bit_sizes=he_config['coeff']
        )
        self.he_ctx.global_scale = he_config['scale']

    def get(self, msg_type: MsgType) -> Msg:

        if msg_type == MsgType.PARTIAL_ENCRYPTED_GRADIENT:
            pass
        elif msg_type == MsgType.VOTE:
            pass
        else:
            raise TypeError(f"Unknown msg_type: {msg_type}")



    def set(self, in_msg: Msg):
        pass

    def run(self):
        """
        Performs local training on the model for the specified number of epochs.

        This method calls the local_train method of the trainer object, passing the number of epochs.
        """
        self.trainer.local_train(self.n_epochs)  # Perform local training for the specified number of epochs