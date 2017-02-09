import os.path as op
import random

import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils.generic_utils import Progbar

from rnn_model import RNNModel
from utils import gen_inflect_from_vocab, dependency_fields


class LanguageModel(RNNModel):

    def __init__(self, *args, **kwargs):
        '''
        field is 'sentence' by default, but can be e.g. pos_sentence.
        '''
        self.field = kwargs.pop('field', 'sentence')
        super(LanguageModel, self).__init__(*args, **kwargs)
        self.class_to_code = {'VBZ': 0, 'VBP': 1}
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        tokens = dep[self.field].split()
        return tokens

    def create_train_and_test(self, examples):
        random.seed(1)
        random.shuffle(examples)

        first = 1
        self.X_train = []
        self.Y_train = []
        self.X_test = self.Y_test = []   # not used; just for compatibility
        self.deps_train = []
        n_train = int(len(examples) * self.prop_train)
        for _, ints, dep in examples[:n_train]:
            self.deps_train.append(dep)
            for i in range(first, len(ints) - 1):
                self.X_train.append(ints[:i])
                self.Y_train.append(ints[i])
            
        self.Y_train = np.asarray(self.Y_train)
        self.deps_test = [x[2] for x in examples[n_train:]]

    def create_model(self):
        self.log('Creating model')
        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab_to_ints) + 1,
                                 self.embedding_size,
                                 input_length=self.maxlen))
        self.model.add(self.rnn_class(output_dim=self.rnn_output_size,
                                      input_length=self.maxlen))
        self.model.add(Dense(len(self.vocab_to_ints) + 1))
        self.model.add(Activation('softmax'))

    def compile_model(self):
        self.log('Compiling model')
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam')

    def results(self):
        recs = []
        columns = ['gram_loss', 'ungram_loss', 'correct'] + dependency_fields
        self.model.model._make_test_function()
        progbar = Progbar(len(self.deps_test))
        for i, dep in enumerate(self.deps_test):
            inp = np.zeros((1, self.maxlen))
            v = int(dep['verb_index']) - 1
            tokens = dep[self.field].split()[:v+1]
            ints = [self.vocab_to_ints[x] for x in tokens]
            try:
                ungram = self.vocab_to_ints[self.inflect_verb[tokens[v]]]
            except KeyError:   # reinflected form not in vocabulary: ignore
                continue
            n = len(ints) - 1
            inp[0, -n:] = ints[:-1]
            gram_loss = self.model.test_on_batch(inp, np.array([ints[v]]))
            ungram_loss = self.model.test_on_batch(inp, np.array([ungram]))
            recs.append((gram_loss, ungram_loss, gram_loss < ungram_loss) +
                        tuple(dep[x] for x in dependency_fields))
            if i % 16 == 0:
                progbar.update(i)

        self.test_results = pd.DataFrame(recs, columns=columns)

    def train(self, n_epochs=10):
        if not hasattr(self, 'model'):
            self.create_model()
            self.compile_model()

        self.serialize_class_data()
        self.serialize_model()

        validation_split = 0.1
        split_at = int(len(self.X_train) * (1. - validation_split))
        x, val_x = self.X_train[:split_at], self.X_train[split_at:]
        y, val_y = self.Y_train[:split_at], self.Y_train[split_at:]
        training_loss_history = []
        validation_loss_history = []

        for epoch in range(n_epochs):
            print 'Epoch', epoch
            training_loss = []
            end = int(float(len(x)) / self.batch_size)
            progbar = Progbar(end)
            for i in range(0, len(x), self.batch_size):
                inp = sequence.pad_sequences(x[i:i+self.batch_size],
                                             maxlen=self.maxlen)
                out = y[i:i+self.batch_size]
                loss = self.model.train_on_batch(inp, out)
                training_loss.append(loss)
                j = int(float(i) / self.batch_size)
                if j % 16 == 0:
                    progbar.update(j)
            progbar.update(end)
                
            # test on validation set
            validation_loss = []
            print
            print 'Evaluating on validation set:'
            end = int(float(len(val_x)) / self.batch_size)
            progbar = Progbar(end)
            for i in range(0, len(val_x), self.batch_size):
                inp = sequence.pad_sequences(val_x[i:i+self.batch_size],
                                             maxlen=self.maxlen)
                out = val_y[i:i+self.batch_size]
                output = self.model.test_on_batch(inp, out)
                validation_loss.append(output)
                j = int(float(i) / self.batch_size)
                if j % 16 == 0:
                    progbar.update(j)
            progbar.update(end)

            training_loss_history.append(np.mean(training_loss))
            validation_loss_history.append(np.mean(validation_loss))
            filename = op.join(self.serialization_dir,
                               'weights_epoch%d.h5' % epoch)
            self.model.save_weights(filename, overwrite=True)
            print
            print ('Mean training loss: %5.3f; mean validation loss: %5.3f\n' %
                   (training_loss_history[-1], validation_loss_history[-1]))
            if (len(validation_loss_history) > 1 and 
                validation_loss_history[-1] >= validation_loss_history[-2]):
                break

        self.training_history = (map(float, training_loss_history),
                                 map(float, validation_loss_history))

    def evaluate(self, howmany=1000):
        self.model.model._make_test_function()
        random.seed(0)
        shuffled = self.deps_test[:]
        random.shuffle(shuffled)
        shuffled = shuffled[:howmany]
        X_test = []
        Y_test = []

        for dep in shuffled:
            tokens = self.process_single_dependency(dep)
            ints = []
            for token in tokens:
                if token not in self.vocab_to_ints:
                    # zero is for pad
                    x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
                    self.ints_to_vocab[x] = token
                ints.append(self.vocab_to_ints[token])

            first = 1
            for i in range(first, len(ints) - 1):
                X_test.append(ints[:i])
                Y_test.append(ints[i])

        test_loss = []
        end = int(float(len(X_test) / self.batch_size))
        progbar = Progbar(end)
        for i in range(0, len(X_test), self.batch_size):
            inp = sequence.pad_sequences(X_test[i:i+self.batch_size],
                                         maxlen=self.maxlen)
            out = Y_test[i:i+self.batch_size]
            output = self.model.test_on_batch(inp, out)
            test_loss.append(output)
            j = int(float(i) / self.batch_size)
            if j % 16 == 0:
                progbar.update(j)
        progbar.update(end)

        return np.mean(test_loss)
