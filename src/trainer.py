import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from src.models.mlp import MLP
from src.models.nac import NAC
from src.models.nalu import NALU
from utils.collections import LOSSES


class Trainer:
    models = {"MLP": MLP,
              "NAC": NAC,
              "NALU": NALU}

    N_EPOCHS_VERBOSENESS = 1

    def __init__(self, lr, model_name, optimizer_name, *model_args, **model_kwargs):
        try:
            model = Trainer.models[model_name]
        except KeyError:
            raise RuntimeError("You have entered an unsupported model type.")
        else:
            self.model = model(*model_args, **model_kwargs)

        try:
            optimizer = LOSSES[optimizer_name]
        except KeyError:
            raise RuntimeError("You have entered an unsupported optimizer.")
        else:
            self.optimizer = optimizer(lr)

    def _loss(self, x, y):
        return tf.losses.mean_squared_error(self.model(x), y)

    def train(self, x, y, x_test=None, y_test=None, n_epochs=100, batch_size=-1, verbose=False):
        batch_size = x.shape[0] if batch_size <= 0 else batch_size
        assert batch_size <= x.shape[0], "Batch size must be less than data shape"
        assert n_epochs > 0, "Must supply a positive number of epochs"

        self.batch_size = batch_size
        self.n_steps = round(x.shape[0] / batch_size)

        for epoch in range(n_epochs):
            print("\nStarting epoch {}...\n".format(epoch))

            # For static arithmetic model, loss is our error measure too
            loss = self._train_epoch(x, y, verbose)

            if epoch % Trainer.N_EPOCHS_VERBOSENESS == 0:
                print("After epoch {} model loss is {}.\n".format(epoch, loss))

                if x_test is not None:
                    test_loss = self._loss(x_test, y_test)
                    print("test loss is {}".format(test_loss))

    def _train_epoch(self, x, y, verbose):
        for step in range(self.n_steps):
            x_batch = x[step*self.batch_size:(step+1)*self.batch_size]
            y_batch = y[step*self.batch_size:(step+1)*self.batch_size]

            loss = self._train_step(x_batch, y_batch)

            if verbose:
                print("At step {} loss is {}".format(step, loss))

        return loss

    def _train_step(self, x_batch, y_batch):
        with tfe.GradientTape() as tape:
            loss = self._loss(x_batch, y_batch)
        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                       global_step=tf.train.get_or_create_global_step())
        return loss


if __name__ == "__main__":
    # Test that trainer works
    print("Testing the Trainer class...")
    tf.enable_eager_execution()

    trainer = Trainer(0.1, "MLP", "Adam", 5, 1, [], act_func="relu")

    x_data = np.random.random((10, 5))
    y_data = np.sum(x_data, axis=1)
    y_data = np.reshape(y_data, (10,1))

    trainer.train(x_data, y_data, verbose=True)
