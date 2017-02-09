import filenames
import os.path as op
import pandas as pd

from google_lm import GoogleLanguageModel
from utils import gen_inflect_from_vocab, get_grouping, annotate_relpron

# deps, glm, inflect_verb = load()
def load():
    glm = GoogleLanguageModel()
    glm.load_affine_transformation()
    deps = pd.read_csv(filenames.deps, delimiter='\t')
    inflect_verb, _ = gen_inflect_from_vocab(filenames.vocab_file)
    return deps, glm, inflect_verb

def process_dep(dep):
    index = dep['verb_index'] - 1
    prefix = dep['orig_sentence'].split()[:index]
    prefix = ' '.join(prefix)
    lstm = glm.process(prefix)
    c = glm.compare_words(lstm, [dep['verb'], inflect_verb[dep['verb']]])
    return c[0] > c[1]

def run_lm_stratified(deps, var, n):
    deps.replace(var, 'na', 0)
    sample = deps.groupby(var).apply(lambda x: x.sample(n))
    sample['correct'] = sample.apply(process_dep, axis=1)
    agged = sample.groupby(var)['correct'].aggregate('mean')
    return agged, sample

def distance():
    short = deps[(deps.distance < 15) & (deps.n_intervening == 0)]
    agged, sample = run_lm_stratified(short, 'distance', 100)
    fn = op.join(filenames.data_dir, 'google_lm', 'distance.csv')
    sample.to_csv(fn)
    return agged, sample

def n_diff_intervening():
    uniform = deps[(deps.n_intervening == deps.n_diff_intervening) &
                   (deps.n_intervening < 5)]
    uniform.replace('n_diff_intervening', 'na', 0)
    agged, sample = run_lm_stratified(uniform, 'n_diff_intervening', 500)
    fn = op.join(filenames.data_dir, 'google_lm', 'n_diff_intervening.csv')
    sample.to_csv(fn)
    return agged, sample

def relprons():
    uniform = deps[(deps.n_intervening == deps.n_diff_intervening) &
                   (deps.n_intervening < 5) & (deps.n_intervening > 0)]
    annotate_relpron(uniform)
    n = 500
    uniform = uniform.groupby('condition').apply(lambda x: x.sample(n))
    uniform = uniform[uniform.condition != 'Error']
    agged, sample = run_lm_stratified(uniform, 'condition', n)
    fn = op.join(filenames.data_dir, 'google_lm', 'relpron.csv')
    sample.to_csv(fn)
    return agged, sample
