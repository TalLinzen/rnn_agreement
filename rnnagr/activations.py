import numpy as np

from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence


class RNNIntrospection(object):

    def __init__(self, rnn_model):
        '''
        rnn_model is expected to be an instance of a RNNModel subclass
        '''
        self.rnn_model = rnn_model
        self.create_truncated_model()

    def create_truncated_model(self):
        self.trunc_model = Sequential()
        emb, rnn = self.rnn_model.model.layers[:2]
        self.trunc_model.add(Embedding(emb.input_dim,
                                       emb.output_dim,
                                       input_length=emb.input_length,
                                       weights=emb.get_weights()))
        self.trunc_model.add(type(rnn)(output_dim=rnn.output_dim,
                                       input_length=rnn.input_length,
                                       weights=rnn.get_weights(),
                                       return_sequences=True,
                                       unroll=True))
        
    def process_sentences(self, sentences):
        ints = [[self.rnn_model.vocab_to_ints[x] for x in sentence.split()]
                for sentence in sentences]
        ints = np.array(ints)
        ints = sequence.pad_sequences(ints, maxlen=self.rnn_model.maxlen)
        classes = self.rnn_model.model.predict_classes(ints)
        classes = [self.rnn_model.code_to_class[x[0]] for x in classes]
        lstm_states = self.trunc_model.predict(ints)
        return classes, lstm_states

    # output of RNN layer is (n_inputs x input_length x n_hidden_units)
