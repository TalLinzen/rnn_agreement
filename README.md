# RNN subject-verb agreement

Code for the paper [Assessing the ability of LSTMs to learn syntax-sensitive
dependencies](https://transacl.org/ojs/index.php/tacl/article/view/972).
Dependencies:

* numpy
* keras
* theano (though the TensorFlow backend is likely to also work)
* pandas

For the Google LM evaluation, you would need to install TensorFlow and download
[the trained model](https://github.com/tensorflow/models/tree/master/lm_1b).

If you're just looking for the subject-verb dependency data in a simple format
and are not planning to run the code in this repository,
download our [simple dependency
dataset](http://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz).

## Quick start

Follow this section if you'd like to run the code on the same set of dependencies we used in the paper.

All of the functions should accept all relevant filenames as arguments, but in
general the easiest thing to do is to set the environment variable
`RNN_AGREEMENT_ROOT` to wherever you cloned this repository.

After cloning the repository, download the [dependency
dataset](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz)
into the `data` subdirectory and unzip the file.

```python
from rnnagr.agreement_acceptor import PredictVerbNumber
from rnnagr import filenames

pvn = PredictVerbNumber(filenames.deps, prop_train=0.1)
pvn.pipeline()
```

After running this code, `pvn.test_results` will be a pandas data frame
with all of the relevant results.

The file `experiments.py` contains the code used to run the experiments
reported in the paper; see that file for examples of tasks and training
regimes other than `PredictVerbNumber`.

## More

If you'd like to regenerate the set of dependencies from the corpus (and
perhaps modify our criteria), download the [subset of the parsed Wikipedia
corpus](http://tallinzen.net/media/rnn_agreement/wikipedia.parsed.subset.50.gz) we used (1.7 GB).

```python
from rnnagr.collect_agreement import CollectAgreement
from rnnagr.utils import deps_to_csv
import filenames
ca = CollectAgreement(filenames.parsed_wiki_subset_50, modes=('infreq_pos',),
skip=0, most_common=10000)
ca.collect_agreement()
deps_to_tsv(ca.deps, 'agr_mostcommon_10K.tsv')
```

## Citation

If you use our data or code for academic work, please cite:

```
@article{linzen2016assessing,
    Author = {Linzen, Tal and Dupoux, Emmanuel and Goldberg, Yoav},
    Journal = {Transactions of the Association for Computational Linguistics},
    Title = {Assessing the ability of {LSTMs} to learn syntax-sensitive dependencies},
    Volume = {4},
    Pages = {521--535},
    Year = {2016}
}
```
