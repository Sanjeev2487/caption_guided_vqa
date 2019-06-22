from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
from pycocotools.coco import COCO
import pickle
import cPickle
from tqdm import tqdm
def create_dictionary(dictionary):
    names = ['train','val']
    for name in names:
        caps = cPickle.load(open( 'VQA_caption_' + name + 'dataset.pkl'))
        for i in tqdm(xrange(len(caps))):
            caps_ = caps[i]['caption']
            for j in xrange(len(caps_)):
                dictionary.tokenize(caps_[j] , True)
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
        vals = map(float, vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

if __name__ == '__main__':

    caption_dictionary = Dictionary()
    caption_dictionary.add_word('<pad>')
    caption_dictionary.add_word('<unk>')
    caption_dictionary = create_dictionary(caption_dictionary)
    caption_dictionary.dump_to_file('caption_dictionary.pkl')
    emb_dim = 300
    glove_file = 'h5data/glove/glove.6B.%dd.txt' % emb_dim
    #with open('/data/wujial/Attention-on-Attention-for-VQA/data/cache/trainval_label2ans.pkl', 'rb') as f:
    #    x = pickle.load(f)
    weights, word2emb = create_glove_embedding_init(caption_dictionary.idx2word, glove_file)
    np.save('glove6b_caption_init_%dd.npy' % emb_dim, weights)
