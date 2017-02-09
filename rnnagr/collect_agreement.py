from collections import Counter
import random

import filenames
from utils import zread, tokenize_blanks, gen_inflect_from_vocab

INDEX = 0
WORD = 1
POS = 3
PARENT_INDEX = -4
LABEL = -3
PARENT = -1


class CollectAgreement(object):

    def __init__(self, infile, modes=('infreq_pos',), most_common=10000,
                 skip=0, stop_after=None, verbose=True, criterion=None,
                 vocab_file=filenames.vocab_file):
        '''
        modes is a tuple of one or more of the following modes:
            'word' - write actual words
            'pos' - replace words with their part of speech
            'infreq_pos' - replace infrequent words with their part of speech
        or None, in which case all modes are produced.

        most_common:
            if mode is 'infreq_pos', only retain this number of words,
            replace the rest with part of speech

        skip:
            number of sentences to skip after each sentence (to avoid all
            sentences starting with the same words if the corpus is sorted)

        criterion:
            None, or function that take a dict representing a dependency and
            returns True if the dependency should be kept
        '''
        self.infile = infile
        self.skip = skip
        self.most_common = most_common
        self.stop_after = stop_after
        self.load_freq_dict(vocab_file)
        self.verbose = verbose
        self.inflect_verb, self.inflect_noun = gen_inflect_from_vocab(
            vocab_file)
        self.criterion = criterion

        allowed_modes = ('word', 'pos', 'infreq_pos')
        self.modes = allowed_modes if modes is None else modes
        if set(self.modes) - set(allowed_modes) != set():
            raise ValueError('Only the following modes are allowed: %s' %
                             allowed_modes)

    def load_freq_dict(self, filename):
        self.freq_dict = Counter()
        for line in file(filename):
            if line.startswith(' '):   # empty string token
                continue
            word, pos, count = line.strip().split()
            word = word.lower()
            self.freq_dict[word] += int(count)
        self.common_words = set(
            dict(self.freq_dict.most_common(self.most_common)).keys())

    def represent_sentence(self, sentence):
        l = [tok[WORD] if tok[WORD] in self.common_words else tok[POS] for
             tok in sentence]
        s = ' '.join(l)
        return s

    def only_nouns(self, sentence, end):
        l = [tok[WORD] if tok[WORD] in self.common_words else tok[POS] for
             tok in sentence if tok[POS] in ['NN', 'NNS'] and
             int(tok[INDEX]) < end]
        s = ' '.join(l)
        return s

    def find_nsubj_agreement(self, sent):
        sentence_dependencies = []
        for tok in sent:
            tok[WORD] = tok[WORD].lower()
            if tok[LABEL] == 'nsubj':
                if tok[POS] not in ['NN', 'NNS']:
                    continue

                if tok[WORD] not in self.inflect_noun:
                    continue

                parent = int(tok[PARENT_INDEX])
                if parent == 0:
                    continue
                # distance from beginning of subject - may not represent
                # the point where number information is encoded
                distance = parent - int(tok[INDEX])
            
                # verify parent does not have an auxiliary
                auxes = [a for a in sent if a[LABEL] == 'aux' and
                         a[PARENT_INDEX] == tok[PARENT_INDEX]]
                if auxes:
                    continue

                parent = sent[parent - 1]
                if (parent[POS] not in ['VBP', 'VBZ'] or
                    parent[WORD] not in self.inflect_verb):
                    continue

                n_intervening = 0
                n_diff_intervening = 0
                max_depth = 0
                last_intervening = 'na'
                middle = sent[int(tok[INDEX]) + 1:int(parent[INDEX]) - 1]
                has_rel = ((int(parent[INDEX]) - int(tok[INDEX]) > 1) and
                           any(x[LABEL] == 'rcmod' for x in middle))
                has_nsubj = ((int(parent[INDEX]) - int(tok[INDEX]) > 1) and
                             any(x[LABEL] == 'nsubj' for x in middle))

                for intervening in sent[int(tok[INDEX]):int(parent[INDEX])]:
                    # This ignores proper nouns (NNP)
                    if intervening[POS] in ['NN', 'NNS']:
                        n_intervening += 1
                        last_intervening = intervening[POS]
                        if intervening[POS] != tok[POS]:
                            n_diff_intervening += 1

                        embedding_depth = 0
                        tmp_node = intervening
                        # Parentheticals can be directly dependent on ROOT (0)
                        # although they are in between the subj and verb
                        while (int(tmp_node[PARENT_INDEX]) not in 
                               (int(tok[INDEX]), 0)):
                            if (tmp_node[POS] in ['NN', 'NNS'] and
                                tmp_node[LABEL] != 'conj'):
                                embedding_depth += 1 
                            tmp_node = sent[int(tmp_node[PARENT_INDEX]) - 1]

                        # Ignoring dependency paths that ended in ROOT
                        if int(tmp_node[PARENT_INDEX]) != 0:
                            max_depth = max(embedding_depth, max_depth)

                subj, verb = tok, parent
                d = {'subj': subj[WORD],
                     'verb': verb[WORD],
                     'subj_pos': subj[POS],
                     'verb_pos': verb[POS],
                     'subj_index': int(subj[INDEX]),
                     'verb_index': int(verb[INDEX]),
                     'n_intervening': n_intervening,
                     'last_intervening': last_intervening,
                     'n_diff_intervening': n_diff_intervening,
                     'distance': distance,
                     'max_depth': max_depth,
                     'has_nsubj': has_nsubj,
                     'has_rel': has_rel}
                if self.criterion is None or self.criterion(d):
                    sentence_dependencies.append(d)
        return sentence_dependencies

    def collect_agreement(self):
        n_deps = 0
        self.deps = []
        random.seed(1)

        if self.verbose and self.stop_after:
            from keras.utils.generic_utils import Progbar
            progbar = Progbar(self.stop_after)

        for i, sent in enumerate(tokenize_blanks(zread(self.infile)), 1):
            if self.stop_after is not None and n_deps >= self.stop_after:
                break
            if i % (self.skip + 1) != 0:
                continue

            # only one dependency per sentence
            deps = self.find_nsubj_agreement(sent)
            if len(deps) == 0:
                continue
            dep = random.choice(deps)
            if dep['subj_index'] > dep['verb_index']:
                continue
            if (dep['subj_pos'] == 'NN' and dep['verb_pos'] == 'VBP' or 
                dep['subj_pos'] == 'NNS' and dep['verb_pos'] == 'VBZ'):
                # ungrammatical dependency (parse error)
                continue

            n_deps += 1
            dep['sentence'] = self.represent_sentence(sent)
            dep['pos_sentence'] = ' '.join(x[POS] for x in sent)
            dep['orig_sentence'] = ' '.join(x[WORD] for x in sent)
            dep['all_nouns'] = self.only_nouns(sent, len(sent))
            dep['nouns_up_to_verb'] = self.only_nouns(sent, 
                                                      int(dep['verb_index']))
            self.deps.append(dep)

            if self.verbose and self.stop_after and n_deps % 10 == 0:
                progbar.update(n_deps)
