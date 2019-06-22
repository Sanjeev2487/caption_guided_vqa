import os
import json
import cPickle
import numpy as np
import butd_utils as utils
import h5py
import torch
from torch.utils.data import Dataset

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    print(w)
                    tokens.append(0)
        return tokens

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


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry

class CaptionQAIMGDataset(Dataset):
    def __init__(self, name, dictionary, caption_dictionary, dataroot='data', caption_dir = 'None'):
        super(CaptionQAIMGDataset, self).__init__()
        assert name in ['train', 'val']
        self.dictionary = dictionary
        self.caption_dictionary = caption_dictionary
        ans2label_path = os.path.join('data', 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join('data', 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.img_id2idx = cPickle.load(open(os.path.join(dataroot, name + '36_imgid2img.pkl' )))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, name + '36.hdf5')
        
        self.hf = h5py.File(h5_path, 'r')
        self.features = self.hf.get('image_features')
        self.spatials = self.hf.get('spatial_features')
        self.entriess  = cPickle.load(open(dataroot + '/VQA_caption_'+name+'dataset.pkl', 'rb'))
        count = 0
        self.entries = {}
        if caption_dir != 'None':
            qid2captions = cPickle.load(open(caption_dir))
        else:
            qid2captions = {}
        for i in xrange(len(self.entriess)):
            for k in xrange(len(self.entriess[i]['question'])):
                new_entry = {'caption': [], 'image': self.entriess[i]['image'], 'question_id' : self.entriess[i]['question'][k]['question_id'], \
                        'question' : self.entriess[i]['question'][k]['question'], 'answer' :  self.entriess[i]['answer'][k] }
                self.entries[count] = new_entry
                self.entries[count]['uid'] = count
                if caption_dir == 'None':
                    for j in xrange(5):
                        self.entries[count]['caption'].append(self.entriess[i]['caption'][j])
                else:
                    self.entries[count]['caption'] = qid2captions[self.entries[count]['question_id']]
                count += 1
                    
        
        self.tokenize()
        self.tensorize()
        self.v_dim = 2048 #self.features.size(2)
        self.s_dim = 36 #self.spatials.size(2)

    def tokenize(self, max_length=14, max_caption_length = 18):
        """Tokenizes the questions.

        """
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            entry['c_token'] =[]
            entry['rc_token'] = []
            for c_id in range(5):
                cap = entry['caption'][c_id]
                caption_tokens = []
                caption_tokens.append(self.caption_dictionary.padding_idx)
                caption_tokens.extend(self.caption_dictionary.tokenize(cap, False))
                caption_tokens.append(self.caption_dictionary.padding_idx)
                caption_tokens = caption_tokens[ : max_caption_length]
                rcaption_tokens = caption_tokens
                if len(caption_tokens) < max_caption_length:
                    # Note here we pad in front of the sentence
                    padding = [self.caption_dictionary.padding_idx] * (max_caption_length - len(caption_tokens))

                    rcaption_tokens = padding + caption_tokens
                    caption_tokens = caption_tokens + padding

                utils.assert_eq(len(caption_tokens), max_caption_length)

                entry['c_token'].append(caption_tokens)
                entry['rc_token'].append(rcaption_tokens[: -1])
            
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self,max_length = 14, max_caption_length = 18):
        #self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            entry['c_token_tensor'] = torch.zeros(5, max_caption_length)
            for c_id in range(5):
                c_token = entry['c_token'][c_id]
                caption = torch.from_numpy(np.array(c_token))
                entry['c_token_tensor'][c_id,: ] = caption

            entry['rc_token_tensor'] = torch.zeros(5, max_caption_length - 1)
            for c_id in range(5):
                c_token = entry['rc_token'][c_id]
                caption = torch.from_numpy(np.array(c_token))
                entry['rc_token_tensor'][c_id,: ] = caption

            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            #entry['uid'] = torch.from_numpy(entry['uid'])
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        imgid = entry['question_id'] // 1000
        img = self.img_id2idx[imgid]
        features = torch.from_numpy(np.array(self.features[img, :, :]))
        features_masks = features #torch.from_numpy(np.array(self.features_masks[entry['image']]))
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        
        if labels is not None:
            target.scatter_(0, labels, scores)
        captions = entry['c_token_tensor']
        rcaptions = entry['rc_token_tensor']
        #uid = entry['uid']
        return features, question, target, captions, rcaptions, features_masks, entry['question_id']

    def __len__(self):
        return len(self.entries)
