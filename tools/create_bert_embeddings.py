import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

import os
import json
import cPickle
import numpy as np
import butd_utils as utils
import h5py
import torch
from torch.utils.data import Dataset
from __future__ import print_function
import os
import sys
import json
import numpy as np
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from dataset import Dictionary
import cPickle


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None, emb_dim=768):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.word2idx = self.tokenizer.vocab
        self.idx2word = ["" for i in range(len(self.word2idx))]
        for key, val in self.tokenizer.vocab.items():
            self.idx2word[val] = key
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        self.word2embedding = {}
        self.weights = torch.zeros((len(self.idx2word), emb_dim), dtype=torch.float32)

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        sentence = self.cls_token + " " + sentence + " " + self.sep_token
        tokenized_words = self.tokenizer.tokenize(sentence)
        #print("Tokenized words: ", tokenized_words)
        #print("Len Tokenized words: ", len(tokenized_words))
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_words)
        segments_ids = [1] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)
        '''
        print("Shape encoded layers: ", encoded_layers.shape)
        count = 0

        for idx in indexed_tokens:
            if idx in self.idx2word:
                self.word2embedding[tokenized_words[count]] += encoded_layers[0][count]
                self.weights[idx] += encoded_layers[0][count]
            count += 1
            # Leaving the case for unknown words
        '''
        return indexed_tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def create_dictionary(dataroot, emb_dim):
    dictionary = Dictionary(emb_dim)
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        #print(path)
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        print("Questions in %s = %d"%(path, len(qs)))
        count = 0
        for q in qs:
            count += 1
            dictionary.tokenize(q['question'], True)
            if count % 100000 == 0:
                print(q)
    return dictionary

emb_dim = 768
d = create_dictionary('data', emb_dim)
d.dump_to_file('data/dictionary.pkl')
weights = d.weights.cpu().detach().numpy()
#word2emb = d.word2embedding.cpu().detach().numpy()
np.save('data/bert_init_%dd.npy' % emb_dim, weights)
