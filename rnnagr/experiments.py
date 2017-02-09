import os.path as op
import os

from keras.layers.recurrent import SimpleRNN
import theano
theano.gof.compilelock.set_lock_status(False)

import filenames
from agreement_acceptor import PredictVerbNumber, CorruptAgreement, \
    PredictVerbNumberOnlyNouns, PredictVerbNumberOnlyGeneralizedNouns, \
    InflectVerb
from language_model import LanguageModel
from collect_agreement import CollectAgreement
from utils import deps_to_tsv

# sentence_subset(filenames.parsed_wiki, '/tmp/newsubset', skip=4)

verbose = 1

def collect_agreement():
    ca = CollectAgreement(filenames.parsed_wiki_subset_50, verbose=verbose,
                          modes=('infreq_pos',), most_common=10000)
    ca.collect_agreement()
    deps_to_tsv(ca.deps, filenames.deps)


def multiple_runs(instance, directory, n_runs=10, start_at=0, verbose=2,
                  debug=False):
    instance.verbose = verbose
    examples = instance.load_examples(1000 if debug else None)
    instance.create_train_and_test(examples)
    fulldir = op.join(filenames.data_dir, directory)
    if not op.exists(fulldir):
        os.mkdir(fulldir)

    for i in range(n_runs):
        run_dirname = op.join(fulldir, '%d' % (i + start_at))
        instance.set_serialization_dir(run_dirname)
        if i == 0:
            instance.serialize_class_data()
        instance.log('Run %d of %d:' % (i + 1, n_runs))
        instance.create_model()
        instance.compile_model()
        instance.train()
        instance.results()
        instance.serialize_model()
        instance.serialize_results()

intervening = lambda dep: dep['n_intervening'] >= 1

models = {
    'grammaticality':
    CorruptAgreement(filenames.deps, prop_train=0.1),
    'predict_number':
    PredictVerbNumber(filenames.deps, prop_train=0.1),
    'language_model':
    LanguageModel(filenames.deps, prop_train=0.1),
    'inflect_verb':
    InflectVerb(filenames.deps, prop_train=0.1),
    'predict_number_targeted':
    PredictVerbNumber(filenames.deps, prop_train=0.2, criterion=intervening),
    'predict_number_only_nouns':
    PredictVerbNumberOnlyNouns(filenames.deps, prop_train=0.1),
    'predict_number_only_generalized_nouns':
    PredictVerbNumberOnlyGeneralizedNouns(filenames.deps, prop_train=0.1),
    'predict_number_srn':
    PredictVerbNumber(filenames.deps, prop_train=0.1, rnn_class=SimpleRNN),
}
