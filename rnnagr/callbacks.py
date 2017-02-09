from keras.callbacks import Callback

class StopWhenLossLow(Callback):

    def __init__(self, threshold=0.01):
        self.threshold = threshold
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if logs['loss'] < self.threshold:
            self.model.stop_training = True
