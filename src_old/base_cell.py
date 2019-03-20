import numpy as np
import tensorflow as tf

class BaseCell:
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}):

        self.x = tf.placeholder(tf.float32, [None, input_dim], name='input')
        self.y = tf.placeholder(tf.float32, [None, output_dim], name='ouput')

        optim = hyper.get('optim', 'rms')
        self.lr = hyper.get('lr', 0.001)

        if optim.lower() == 'adam':
            _optim = tf.train.AdamOptimizer(self.lr)
        elif optim.lower() == 'gd':
            _optim = tf.train.GradientDescentOptimizer(self.lr)
        elif optim.lower() == 'rms':
            _optim = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise NotImplementedError("Learning algorithm not recognised")

        self.optim = _optim

    def train(self, x, y, x_test=None, y_test=None, N_epochs=10000, batchSize=1000):
        if not batchSize:
            batchSize = x.shape[0]

        self.batchSize = batchSize
        self.n_steps = round(x.shape[0] / batchSize)

        old_loss = np.inf
        small_loss_count = 1000
        nan_count = 100
        for epoch in range(N_epochs):
            err = self._train_epoch(x, y)
            if epoch % 100 == 0:
                print("\nEpoch: {}".format(epoch))
                print("Err: {}".format(err))
            if epoch % 200 == 0 and x_test is not None:
                err_test, y_hat = self.validate(x_test, y_test)
                print("Test error: {}".format(err_test))
            if old_loss - err < .00001:
                small_loss_count -= 1
                if small_loss_count < 0:
                    print("Early Stopping after {} epochs of no improvements".format(1000))
                    break
            else:
                small_loss_count = 1000
            if np.isnan(err):
                nan_count -= 1
                if nan_count < 0:
                    print("Early Stopping after 100 epochs of NaNs")
                    break
            else:
                nan_count = 100
            old_loss = err

        print("Final training error: {}".format(err))
        err_test, y_hat = self.validate(x_test, y_test)
        print("Final test error: {}".format(err_test))

        return err_test

    def _train_epoch(self, x, y):
        for step in range(self.n_steps):
            if step < self.n_steps - 1:
                x_step = x[step*self.batchSize:(step+1)*self.batchSize]
                y_step = y[step*self.batchSize:(step+1)*self.batchSize]
            else:
                x_step = x[step*self.batchSize:]
                y_step = y[step*self.batchSize:]

            err = self._train_step(x_step, y_step)

        return err

    def _train_step(self, x_step, y_step):
            feed_dict = {self.x: x_step, self.y: y_step}

            ops = [self.loss, self.optimise]
            loss, _ = self._Sess.run(ops, feed_dict)
            return loss

    def validate(self, x, y):
        feed_dict = {self.x: x, self.y: y}
        ops = [self.loss, self.y_hat]
        loss, y_hat = self._Sess.run(ops, feed_dict)

        return loss, y_hat
