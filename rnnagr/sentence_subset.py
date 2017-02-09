from utils import zread

total_n_sentences = 77633994.0

def read_file(fh):
    sent = []
    for line in fh:
        line = line.strip()
        if not line:
            if sent:
                yield sent
            sent = []
        else:
            sent.append(line)
    yield sent


def sentence_subset(infile, outfile, skip, maxlen=50):
    '''
    Extracts a subset of sentences from a gziped file (e.g. every fourth
    sentence).
    '''

    out = open(outfile, 'w')
    n = 0

    for i, sent in enumerate(read_file(zread(infile)), 1):
        if i % (skip + 1) != 0 or len(sent) > maxlen:
            continue
        if i % 100000 == 0:
            print '%5.3f %d' % (i / total_n_sentences, n)
        out.write('\n'.join(sent) + '\n\n')
        n += 1
