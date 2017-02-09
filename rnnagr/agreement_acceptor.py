import random

from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
import pandas as pd

from rnn_model import RNNModel
from utils import gen_inflect_from_vocab, dependency_fields


class RNNAcceptor(RNNModel):

    def create_train_and_test(self, examples):
        d = [[], []]
        for i, s, dep in examples:
            d[i].append((i, s, dep))
        random.seed(1)
        random.shuffle(d[0])
        random.shuffle(d[1])
        if self.equalize_classes:
            l = min(len(d[0]), len(d[1]))
            examples = d[0][:l] + d[1][:l]
        else:
            examples = d[0] + d[1]
        random.shuffle(examples)

        Y, X, deps = zip(*examples)
        Y = np.asarray(Y)
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        n_train = int(self.prop_train * len(X))
        self.X_train, self.Y_train = X[:n_train], Y[:n_train]
        self.X_test, self.Y_test = X[n_train:], Y[n_train:]
        self.deps_train = deps[:n_train]
        self.deps_test = deps[n_train:]

    def create_model(self):
        self.log('Creating model')
        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab_to_ints) + 1,
                                 self.embedding_size,
                                 input_length=self.maxlen))
        self.model.add(self.rnn_class(output_dim=self.rnn_output_size,
                                      input_length=self.maxlen,
                                      unroll=True))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    def compile_model(self):
        self.log('Compiling model')
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.Y_test,
                                   batch_size=self.batch_size)

    def results(self):
        self.log('Processing test set')
        predicted = self.model.predict_classes(self.X_test,
                                               verbose=self.verbose).flatten()
        recs = []
        columns = ['correct', 'prediction', 'label'] + dependency_fields
        for dep, prediction in zip(self.deps_test, predicted):
            prediction = self.code_to_class[prediction]
            recs.append((prediction == dep['label'], prediction, 
                         dep['label']) +
                        tuple(dep[x] for x in dependency_fields))
        self.test_results = pd.DataFrame(recs, columns=columns)


class CorruptAgreement(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        tokens = dep['sentence'].split()
        if random.random() < 0.5:
            dep['label'] = 'ungrammatical'
            v = int(dep['verb_index']) - 1
            tokens[v] = self.inflect_verb[tokens[v]]
            dep['sentence'] = ' '.join(tokens)
        else:
            dep['label'] = 'grammatical'
        return tokens


class PredictVerbNumber(RNNAcceptor):

    def __init__(self, *args, **kwargs):
        RNNAcceptor.__init__(self, *args, **kwargs)
        self.class_to_code = {'VBZ': 0, 'VBP': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        v = int(dep['verb_index']) - 1
        tokens = dep['sentence'].split()[:v]
        return tokens


class PredictVerbNumberOnlyNouns(PredictVerbNumber):

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        tokens = dep['nouns_up_to_verb'].split()
        return tokens


class PredictVerbNumberOnlyGeneralizedNouns(PredictVerbNumber):

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        tokens = dep['sentence'].split()[:dep['verb_index']]
        poses =  dep['pos_sentence'].split()[:dep['verb_index']]
        tokens = [token for token, pos in zip(tokens, poses) if
                  pos in ['NN', 'NNS', 'NNP', 'PRP']]
        print dep['sentence']
        print tokens
        print
        return tokens


class InflectVerb(PredictVerbNumber):
    '''
    Present all words up to _and including_ the verb, but withhold the number
    of the verb (always present it in the singular form). Supervision is
    still the original number of the verb. This task allows the system to use
    the semantics of the verb to establish the dependency with its subject, so
    may be easier. Conversely, this may mess up the embedding of the singular
    form of the verb; one solution could be to expand the vocabulary with
    number-neutral lemma forms.
    '''

    def __init__(self, *args, **kwargs):
        super(InflectVerb, self).__init__(*args, **kwargs)
        self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

    def process_single_dependency(self, dep):
        dep['label'] = dep['verb_pos']
        v = int(dep['verb_index']) - 1
        tokens = dep['sentence'].split()[:v+1]
        if dep['verb_pos'] == 'VBP':
            tokens[v] = self.inflect_verb[tokens[v]]
        return tokens
