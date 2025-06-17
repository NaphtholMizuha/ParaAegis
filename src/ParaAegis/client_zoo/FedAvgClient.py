from BaseClient import BaseClient
from BaseClient import MsgType
from BaseClient import Msg
from ..trainer import Trainer
import msgpack

class FedAvgClient(BaseClient):
    """
    FedAvgClient is a class that implements the Federated Averaging (FedAvg) client.
    It inherits from the BaseClient class and provides methods for message handling and local training.
    """

    def __init__(
        self,
        in_msg_type: MsgType,
        out_msg_type: MsgType,
        trainer: Trainer,
        n_epochs: int
    ):
        """
        Initializes the FedAvgClient with input and output message types, a trainer object, and the number of training epochs.

        :param in_msg_type: The type of incoming messages expected by the client.
        :param out_msg_type: The type of outgoing messages produced by the client.
        :param trainer: The trainer object responsible for training the model.
        :param n_epochs: The number of epochs to train the model locally.
        """
        super().__init__(in_msg_type, out_msg_type)
        self.trainer = trainer  # Initialize the trainer object
        self.n_epochs = n_epochs  # Set the number of training epochs

    def get(self) -> Msg:
        """
        Retrieves the current gradient from the trainer and returns it as a serialized message.

        :return: A Msg object containing the serialized gradient.
        """
        grad = self.trainer.get_grad()  # Get the gradient from the trainer
        grad = msgpack.dumps(grad)  # Serialize the gradient using msgpack
        return Msg([grad], MsgType.GRADIENT)  # Return the serialized gradient as a message

    def set(self, in_msg: Msg):
        """
        Processes an incoming message containing a gradient and sets it in the trainer.

        :param in_msg: The incoming message containing the gradient.
        :raises TypeError: If the message type is not supported.
        """
        if in_msg.type == MsgType.GRADIENT:
            grad = msgpack.loads(in_msg.data)  # Deserialize the gradient from the message
            self.trainer.set_grad(grad)  # Set the gradient back to the trainer
        else:
            raise TypeError(f"FedAvgClient cannot process type: {in_msg.type}")  # Raise an error for unsupported message types

    def run(self):
        """
        Performs local training on the model for the specified number of epochs.

        This method calls the local_train method of the trainer object, passing the number of epochs.
        """
        self.trainer.local_train(self.n_epochs)  # Perform local training for the specified number of epochs