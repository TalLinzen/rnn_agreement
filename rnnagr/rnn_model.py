import json
import multiprocessing
import os
import os.path as op
import cPickle as pickle
import random

from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import LSTM
import pandas as pd

import filenames
from utils import deps_from_tsv


class RNNModel(object):

    serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
                             'X_train', 'Y_train', 'deps_train',
                             'X_test', 'Y_test', 'deps_test']

    def __init__(self, filename=None, serialization_dir=None,
                 batch_size=16, embedding_size=50, 
                 maxlen=50, prop_train=0.9, rnn_output_size=50,
                 mode='infreq_pos', vocab_file=filenames.vocab_file,
                 rnn_class=LSTM, equalize_classes=False, criterion=None,
                 verbose=1):
        '''
        filename: TSV file with positive examples, or None if unserializing
        criterion: dependencies that don't meet this criterion are excluded
            (set to None to keep all dependencies)
        verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
        '''
        self.filename = filename
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.prop_train = prop_train
        self.mode = mode
        self.rnn_output_size = rnn_output_size
        self.rnn_class = rnn_class
        self.maxlen = maxlen
        self.equalize_classes = equalize_classes
        self.criterion = (lambda x: True) if criterion is None else criterion
        self.verbose = verbose
        self.set_serialization_dir(serialization_dir)

    def log(self, message):
        print message

    def pipeline(self):
        examples = self.load_examples()
        self.create_train_and_test(examples)
        self.create_model()
        self.compile_model()
        self.train()
        self.results()

    def load_examples(self, n_examples=None):
        '''
        Set n_examples to some positive integer to only load (up to) that 
        number of examples
        '''
        self.log('Loading examples')
        if self.filename is None:
            raise ValueError('Filename argument to constructor can\'t be None')

        self.vocab_to_ints = {}
        self.ints_to_vocab = {}
        examples = []
        n = 0

        deps = deps_from_tsv(self.filename, limit=n_examples)

        for dep in deps:
            tokens = dep['sentence'].split()
            if len(tokens) > self.maxlen or not self.criterion(dep):
                continue

            tokens = self.process_single_dependency(dep)
            ints = []
            for token in tokens:
                if token not in self.vocab_to_ints:
                    # zero is for pad
                    x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
                    self.ints_to_vocab[x] = token
                ints.append(self.vocab_to_ints[token])

            examples.append((self.class_to_code[dep['label']], ints, dep))
            n += 1
            if n_examples is not None and n >= n_examples:
                break

        return examples

    def set_serialization_dir(self, d):
        if d is None:
            d = '/tmp/%s_%d' % (self.__class__.__name__,
                                random.randint(0, 100))
        if not op.exists(d):
            os.mkdir(d)
        self.serialization_dir = d

    def serialize_class_data(self):
        class_data = {x: getattr(self, x) for x in self.serialized_attributes}
        filename = op.join(self.serialization_dir, 'class_data.pickle')
        pickle.dump(class_data, open(filename, 'w'))

    def serialize_model(self):
        model_string = self.model.to_json()
        model_filename = op.join(self.serialization_dir, 'model.json')
        weights_filename = op.join(self.serialization_dir, 'weights.h5')
        open(model_filename, 'w').write(model_string)
        self.model.save_weights(weights_filename, overwrite=True)

    def serialize_results(self):
        filename = op.join(self.serialization_dir, 'test_results.csv')
        self.test_results.to_csv(filename)
        filename = op.join(self.serialization_dir, 'training_history.json')
        json.dump(self.training_history, open(filename, 'w'))

    def serialize(self):
        self.serialize_class_data()
        self.serialize_model()
        self.serialize_results()

    def unserialize(self, d, epoch=None, load_everything=True):
        self.model = model_from_json(open(op.join(d, 'model.json')).read())
        weights_file = ('weights.h5' if epoch is None else
                        'weights_epoch%d.h5' % epoch)
        self.model.load_weights(op.join(d, weights_file))

        if load_everything:
            class_data_file = op.join(d, 'class_data.pickle')
            if op.exists(class_data_file):
                class_data = pickle.load(open(class_data_file))
                for x in self.serialized_attributes:
                    setattr(self, x, class_data[x])

            test_results = op.join(d, 'test_results.csv')
            if os.path.exists(test_results):
                self.test_results = pd.read_csv(test_results)

            training_history = op.join(d, 'training_history.json')
            if os.path.exists(training_history):
                self.training_history = json.load(open(training_history))

    def train(self, n_epochs=10):
        self.log('Training')
        if not hasattr(self, 'model'):
            self.create_model()
            self.compile_model()
        es = EarlyStopping(monitor='val_loss')
        history = self.model.fit(self.X_train, self.Y_train,
                                 validation_split=0.1,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose,
                                 nb_epoch=n_epochs, callbacks=[es])
        self.training_history = history.history
