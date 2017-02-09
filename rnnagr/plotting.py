from IPython.core.display import display, HTML
import matplotlib.pylab as plt
import matplotlib
import pandas as pd

from utils import get_grouping

# Not used in this file, but useful for jupyter notebooks
import os.path as op
import filenames
import seaborn as sns

matplotlib.style.use('ggplot')
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.autolayout'] = True

task_pretty_names = [
    ('predict_number_only_nouns', 'Number prediction baseline (common nouns)'),
    ('predict_number_only_generalized_nouns', 'Number prediction baseline (all nouns)'),
    ('predict_number', 'Number prediction (LSTM)'),
    ('predict_number_ensemble', 'Number prediction (LSTM: ensemble)'),
    ('predict_number_srn', 'Number prediction (SRN)'),
    ('predict_number_targeted', 'Number prediction (LSTM: targeted)'),
    ('grammaticality', 'Grammaticality judgments (LSTM)'),
    ('grammaticality_targeted', 'Grammaticality judgments (LSTM: targeted)'),
    ('inflect_verb', 'Verb inflection (LSTM)'),
    ('language_model', 'Language modeling (LSTM)'),
]
task_pretty_names_dict = dict(task_pretty_names)

task_shortcut_names = [
    ('predict_number_only_nouns', 'Baseline (common nouns)'),
    ('predict_number_only_generalized_nouns', 'Baseline (all nouns)'),
    ('predict_number', 'NumPred'),
    ('predict_number_srn', 'NumPred (SRN)'),
    ('predict_number_targeted', 'NumPredNP (targeted)'),
    ('grammaticality', 'GramJudg'),
    ('grammaticality_targeted', 'GramJudg (targeted)'),
    ('inflect_verb', 'VerbInfl'),
    ('language_model', 'LangMod'),
]
task_shortcut_names_dict = dict(task_shortcut_names)

def highlight_dep(dep, show_pos=False):
    s = []
    s2 = []
    z = zip(dep.orig_sentence.split(), dep.sentence.split(),
            dep.pos_sentence.split())
    for i, (tok, mixed, pos) in enumerate(z):
        color = 'black'
        if i == dep.subj_index - 1 or i == dep.verb_index - 1:
            color = 'blue'
        elif (dep.subj_index - 1 < i < dep.verb_index - 1 and 
              pos in ['NN', 'NNS']): 
            color = 'red' if pos != dep.subj_pos else 'green'
        s.append('<span style="color: %s">%s</span>' % (color, tok))
        if show_pos:
            s2.append('<span style="color: %s">%s</span>' % (color, pos))
    res = ' '.join(s)
    if show_pos:
        res +=  '<br>' + ' '.join(s2)
    display(HTML(res))


def clean_ticks():
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def percent_ylabel():
    plt.gca().set_yticklabels(['%d%%' % (x * 100) for
                               x in plt.gca().get_yticks()])

def eb(x, field, **kwargs):
    yerr = [x['errorprob'] - x['minconf'], x['maxconf'] - x['errorprob']]
    plt.errorbar(x[field], x['errorprob'], yerr=yerr, **kwargs)

def errorplot(x, y, minconf, maxconf, **kwargs):
    '''
    e.g.
    g = sns.FacetGrid(attr, col='run', hue='subj_pos', col_wrap=5)
    g = g.map(errorplot, 'n_diff_intervening', 'errorprob',
        'minconf', 'maxconf').add_legend()
    '''
    plt.errorbar(x, y, yerr=[y - minconf, maxconf - y], fmt='o-', **kwargs)


def add_relclause_annotation(res):
    uniform = res[res.n_diff_intervening == res.n_intervening]
    def f(x):
        blacklist = set(['NNP', 'PRP'])
        relprons = set(['WDT', 'WP', 'WRB', 'WP$'])
        words_in_dep = x['orig_sentence'].split()[x['subj_index']:x['verb_index']-1]
        pos_in_dep = x['pos_sentence'].split()[x['subj_index']:x['verb_index']-1]
        first_is_that = words_in_dep[:1] == ['that']
        return (bool(blacklist & set(pos_in_dep)), 
                bool(relprons & set(pos_in_dep[:2])) | first_is_that,
                bool(relprons & set(pos_in_dep)) | first_is_that)

    uniform['blacklisted'], uniform['has_early_relpron'], uniform['has_relpron'] = zip(*uniform.apply(f, axis=1))

    pd.options.mode.chained_assignment = None
    df = uniform[~uniform.blacklisted]
    rel_groups = get_grouping(df[df.n_intervening < 4],
                             ['n_intervening', 'has_rel', 'has_relpron', 'has_early_relpron'])
    rel_groups = rel_groups.reset_index()

    def g(x):
        if x['has_rel'] and x['has_relpron'] and x['has_early_relpron']:
            return 'Rel with early pronoun'
        elif x['has_rel'] and x['has_relpron'] and not x['has_early_relpron']:
            return 'Rel with late pronoun'
        elif x['has_rel'] and not x['has_relpron']:
            return 'Rel without pronoun'
        elif not x['has_rel']:
            if x['has_relpron']:
                return 'Error'
            else:
                return 'No rel'
        else:
            return 'Error'
        
    rel_groups['condition'] = rel_groups.apply(g, axis=1)
    rel_groups = rel_groups[rel_groups.condition != 'Error']
    rel_groups[['n_intervening', 'condition', 'count', 'errorprob']]
    return rel_groups

