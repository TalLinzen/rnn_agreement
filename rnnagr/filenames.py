import os
import os.path as op

root = os.environ['RNN_AGREEMENT_ROOT']

data_dir = op.join(root, 'data')
parsed_wiki = op.join(data_dir, 'wikipedia.wcase.nodups.parsed.fixed.gz')
parsed_wiki_subset = op.join(data_dir, 'wikipedia.parsed.subset.50.gz')
vocab_file = op.join(data_dir, 'wiki.vocab')
deps = op.join(data_dir, 'agr_50_mostcommon_10K.tsv')
figures = op.join(root, 'writeups', 'figures')
overall_report = op.join(data_dir, 'overall_report.csv')
google_lm_dir = op.join(root, 'tf', 'data')
