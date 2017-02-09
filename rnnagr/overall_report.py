import os
import os.path as op

import pandas as pd

import filenames
from utils import get_grouping


class OverallReport(object):

    language_model = ['language_model', 'language_model_100',
                      'language_model_200', 'language_model_500']

    all_tasks = [
        'predict_number', 'predict_number_targeted', 'predict_number_srn',
        'predict_number_srn_quadruple', 'predict_number_only_nouns',
        'inflect_verb', 'grammaticality', 'grammaticality_targeted',
        'predict_number_only_generalized_nouns'
    ] + language_model

    def __init__(self, tasks=None, join=False, deps=None, debug=False): 
        self.join = join
        self.debug = debug
        self.tasks = tasks if tasks is not None else self.all_tasks
        self.nrows = None if debug is False else 1000
        self.ensemble_dir = op.join(filenames.data_dir, 'ensemble')
        if not op.exists(self.ensemble_dir):
            os.mkdir(self.ensemble_dir)

        if join:
            if deps is None:
                self.deps = pd.read_table(filenames.deps, nrows=self.nrows)
            else:
                self.deps = deps

    def groupings(self, results, df, task, run):
        g = get_grouping(df, ('subj_pos', 'last_intervening'))
        results.setdefault('last', []).append(g)

        uniform = df[(df.n_intervening == df.n_diff_intervening) &
                     (df.n_intervening < 5)]
        uniform.replace('n_diff_intervening', 'na', 0)
        g = get_grouping(uniform, ('subj_pos', 'n_diff_intervening'))
        results.setdefault('n_attractors', []).append(g)

        g = get_grouping(uniform, ('n_diff_intervening',))
        results.setdefault('n_attractors_collapsed', []).append(g)

        if 'targeted' not in task:
            no_intervening = df[(df.n_intervening == 0) & (df.distance < 15)]
            g = get_grouping(no_intervening, ('distance',))
            results.setdefault('only_distance', []).append(g)

        for x in results.values():
            x[-1]['run'] = run

    def load_task_results(self, task, stop_after=None):
        d = op.join(filenames.data_dir, task)
        l = os.listdir(d)
        results = {}
        end = min(stop_after, len(l)) if stop_after is not None else len(l)
        for i, s in enumerate(l):
            if stop_after is not None and i == stop_after:
                break
            print '%d out of %d' % (i + 1, end)
            print '   Reading results'
            df = pd.read_csv(op.join(d, s, 'test_results.csv'),
                             nrows=self.nrows)
            if self.join:
                df = pd.merge(df, self.deps)
            if i == 0:
                ensemble = df
                ensemble['n_correct'] = 0

            ensemble['n_correct'] += df['correct'].astype(int)
            if (ensemble['orig_sentence'].head(100) !=
                df['orig_sentence'].head(100)).any():
                print 'Run %d: Sentences not aligned across runs' % i

            print '   Calculating summaries'
            self.groupings(results, df, task, s)

        print 'Generating ensemble'
        ensemble['correct'] = ensemble.n_correct.divide(len(l)) > 0.5
        self.groupings(results, ensemble, task, 'ensemble')

        concs = []
        for grouping, dfs in results.items():
            conc = pd.concat(dfs)
            conc['grouping'] = grouping
            concs.append(conc)

        ensemble.to_csv(op.join(self.ensemble_dir, '%s.csv' % task))

        return pd.concat(concs)

    def generate_overall_report(self, filename=None):
        dfs = []
        tasks = self.tasks[:1] if self.debug else self.tasks
        limit = 3 if self.debug else None
        for task in tasks:
            print '\n%s\n%s' % (task, '=' * len(task))
            df = self.load_task_results(task, limit)
            df['task'] = task
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv(filename or filenames.overall_report)
