import random

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

import filenames


def create_pos_dict(vocab_file):
    freq_threshold = 50
    unambig_threshold = 0.9

    by_pos = {}
    for line in file(vocab_file):
        if line.startswith(' '):   # empty string token
            continue
        word, pos, count = line.strip().split()
        if word[0].isupper():
            continue
        count = int(count)
        if len(word) > 1 and count >= freq_threshold:
            by_pos.setdefault(word, {})[pos] = count
    
    pos_unambig_words = {}
    for word, pos_freqs in by_pos.items():
        total = sum(pos_freqs.values())
        for key, value in pos_freqs.items():
            if float(value) / total > unambig_threshold:
                pos_unambig_words[word] = key
                break

    return pos_unambig_words


def visualize_embeddings(aa, pca=True, text=True, n=100, components=(0, 1),
                         relevant_pos=['NN', 'NNS']):
    pos_dict = create_pos_dict(filenames.vocab_file)
    c1, c2 = components
    embeddings = aa.model.layers[0].get_weights()[0]
    pos_tags = ['NN', 'NNS', 'VBZ', 'VB']
    pos_pretty = ['Singular noun', 'Plural noun', 'Singular verb',
                  'Base verb form']

    pos_in_vocab = {}
    for x in aa.vocab_to_ints.keys():
        pos = pos_dict.get(x)
        if pos in relevant_pos:
            pos_in_vocab.setdefault(pos, []).append(x)
    
    samples = []
    random.seed(1)
    for p in relevant_pos:
        tmp = pos_in_vocab[p]
        random.shuffle(tmp)
        samples += [(p, x) for x in tmp[:n]]

    mat = np.array([embeddings[aa.vocab_to_ints[x[1]]] for x in samples])
    if pca:
        pca = PCA()
        mat = pca.fit(embeddings).transform(mat)

    plt.scatter(mat[:, c1], mat[:, c2], c='white', edgecolors='white')
    colors = ['red', 'blue', 'green', 'black']
    colordict = dict(zip(pos_tags, colors))
    if text:
        for i, (p, word) in enumerate(samples):
            color = colordict[p]
            plt.annotate(word, (mat[i, c1], mat[i, c2]), color=color)
            #if p == 'NNS' and mat[i, c1] < 0.5 or p == 'NN' and mat[i, c1] > 0.5:
            #    print word
    else:
        x = {}
        y = {}
        for i, (p, word) in enumerate(samples):
            x.setdefault(p, []).append(mat[i, c1])
            y.setdefault(p, []).append(mat[i, c2])
        plots = []
        for p in relevant_pos:
            plots.append(plt.scatter(x[p], y[p], color=colordict[p], alpha=0.3))
            legend_items = [yy for xx, yy in zip(pos_tags, pos_pretty) if 
                            xx in relevant_pos]
            plt.legend(plots, legend_items, loc='lower left')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    return plt
