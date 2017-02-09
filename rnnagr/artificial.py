import random

from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from utils import StopWhenLossLow

class ToyData(object):

    vocab_size = 10

    def __init__(self, n_train, n_test):
        self.n_train = n_train
        self.n_test = n_test
    
    def rand_subj_verb_pair(self, grammatical):
        subject = random.randint(1, 2)
        if grammatical:
            verb = [3, 4][subject - 1]
        else:
            verb = [4, 3][subject - 1]
        return (subject, verb)

    def fillers(self, min_len, max_len):
        l = random.randint(min_len, max_len)
        return [random.randint(5, 9) for i in range(l)]

    def train(self, rnn_class=LSTM, n_epochs=10):
        random.shuffle(self.examples)

        embedding_size = 20
        rnn_output_size = 20
        batch_size = 16

        Y = [i[0] for i in self.examples]
        X = sequence.pad_sequences([i[1] for i in self.examples], 
                                   maxlen=self.maxlen)
        X_train, Y_train = X[:self.n_train], Y[:self.n_train]
        X_test, Y_test = X[self.n_train:], Y[self.n_train:]

        model = Sequential()
        model.add(Embedding(self.vocab_size, embedding_size,
                            input_length=self.maxlen))
        model.add(rnn_class(output_dim=rnn_output_size,
                            input_length=self.maxlen))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'],
                      class_mode='binary')
        model.summary()

        cb = StopWhenLossLow()

        history = model.fit(X_train, Y_train, 
                            batch_size=batch_size, 
                            nb_epoch=n_epochs,
                            validation_split=0.1,
                            callbacks=[cb])
        score = model.evaluate(X_test, Y_test, batch_size=batch_size,
                               show_accuracy=True)
        return history, model, score


class LinearToyData(ToyData):
    '''
    If 1 is the "subject", the "verb" 3 must occur at some later point;
    same for 2 and 4. Variable length sequences of 5 through 9 occur before
    the "subject", between the "subject" and the "verb", and after the "verb".
    '''

    def __init__(self, *args):
        ToyData.__init__(self, *args)
        self.maxlen = 10

    def generate_data(self):
        self.examples = []
        for c in [0, 1]:
            for i in range((self.n_train + self.n_test) / 2):
                subject, verb = self.rand_subj_verb_pair(1 - c)
                sentence = (self.fillers(1, 2) +
                            [subject] +
                            self.fillers(1, 2) + 
                            [verb] +
                            self.fillers(1, 2))
                self.examples.append((c, sentence))
                

class HierarchicalToyData(ToyData):

    def __init__(self, *args):
        ToyData.__init__(self, *args)
        self.maxlen = 14
    
    def generate_data(self):
        self.examples = []
        for c in [0, 1]:
            for i in range((self.n_train + self.n_test) / 2):
                main_grammatical, sub_grammatical = True, True
                if c == 1:
                    main_grammatical = bool(random.randint(0, 1))
                    sub_grammatical = not main_grammatical
                main_subject, main_verb = self.rand_subj_verb_pair(
                    main_grammatical)
                sub_subject, sub_verb = self.rand_subj_verb_pair(
                    sub_grammatical)
                sentence = (self.fillers(1, 2) +
                            [main_subject] +
                            self.fillers(1, 2) +
                            [sub_subject] +
                            self.fillers(1, 2) +
                            [sub_verb] +
                            self.fillers(1, 2) +
                            [main_verb] +
                            self.fillers(1, 2))
                self.examples.append((c, sentence))


