from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
import pickle as cPickle

def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'VQA_caption_traindataset.pkl',
        'VQA_caption_valdataset.pkl'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        dataset = cPickle.load(open(question_path, 'rb'))
        for idx in range(len(dataset)):
            captions = dataset[idx]['caption']
            for cap in captions:
                dictionary.tokenize(cap, True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        #vals = map(float, vals[1:])
        valv = [float(v) for v in vals[1:]]
        word2emb[word] = np.array(valv)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d = create_dictionary('data')
    d.dump_to_file('data/caption_dictionary.pkl')

    d = Dictionary.load_from_file('data/caption_dictionary.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('data/glove6b_caption_init_%dd.npy' % emb_dim, weights)
