from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
import eval_utils
from misc.utils import *

from dataset import *
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def decode_caption(idx2word, seqs):
    seqs = seqs.data.detach().cpu().numpy()
    captions = []
    for i in range(len(seqs)):
        caption = ''
        for j in range(seqs.shape[1]):
            idx = seqs[i, j]
            if idx == len(idx2word):
                break
            else:
                caption = caption + idx2word[idx] + ' '

        captions.append(' '.join(caption.split()))
    return captions


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def evaluate(model, dataloader, with_gt_caption= 0):
    score = 0
    num_data = 0
    model.eval()
    for v, q, a, c, rc, vm, _ in iter(dataloader):
        v = v.cuda()
        q = q.cuda()
        a = a.cuda() # boxes not used
        c = c.long().cuda()
        rc = rc.long().cuda()
        pred = model(v, q, c, rc, with_gt_caption, 'VQA')
        batch_score = compute_score_with_logits( pred, a.data).sum()
        score += batch_score
        num_data += pred.size(0)

    score /= num_data
    model.train()
    return score

def weights_init_kn(m):
     if isinstance(m, nn.Linear):
         nn.init.kaiming_normal(m.weight.data, a=0.01)

def evaluate_caption(model, dataloader, idx2word, step = 10000, beam_search = 1):
    score = 0
    num_data = 0
    model.eval()
    qid2caption = {}
    i = 0
    for v, q, a, c, rc, vm,  qids in tqdm(iter(dataloader)):
        v = v.cuda()
        q = q.cuda()
        a = a.cuda() # boxes not used
        c = c.long().cuda()
        rc = rc.long().cuda()
        if beam_search == 0:
            seq, seqLogprobs = model._sample(v, q, c, rc)
        else:
            _, done_beams = model._sample_beam(v, q, c, rc)
            seq = []
            for b in range(len(done_beams)):
                #for bm in range(5):
                #    seq.append(done_beams[b][bm * 2]['seq'])
                seq.append(done_beams[b][0]['seq'])
            seq = torch.stack(seq)
        captions = decode_caption(idx2word, seq)
        for j in range(len(qids)):
            #qid2caption[int(qids[j])] = captions[j * 5: (j + 1) * 5]
            qid2caption[int(qids[j])] = captions[j : j + 1]
            print(qid2caption[int(qids[j])])
        if i == step:
            break
        else:
            i += 1
        
    model.train()
    return qid2caption

def train(opt):
    # Deal with feature things before anything
    crit = LanguageModelCriterion()
    dictionary = Dictionary.load_from_file('data/dictionary.pkl') # question dictionary
    caption_dictionary = Dictionary.load_from_file('data/caption_dictionary.pkl')
    train_dset = CaptionQAIMGDataset('train', dictionary, caption_dictionary)
    actual_batch_size = 32
    train_loader = DataLoader(train_dset, actual_batch_size, shuffle=False, num_workers=0)
    eval_dset = CaptionQAIMGDataset('val', dictionary, caption_dictionary)
    eval_loader  = DataLoader(eval_dset, actual_batch_size, shuffle=False, num_workers=0)
    opt.vocab_size = len(caption_dictionary.word2idx)
    opt.n_tokens = len(dictionary.word2idx)
    opt.seq_length = 17 
    model = models.setup(opt).cuda()
    model.apply(weights_init_kn)

    model.train()

    opt.checkpoint_path = 'saved_models/%s' % (
        str(datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_'))
    os.mkdir(opt.checkpoint_path)
    log_file = open(opt.checkpoint_path + '/log.txt', 'w')
    print(opt, file=log_file)
    log_file.flush()

    epoch = opt.epoch

    model.load_state_dict( torch.load("saved_models/2019_06_19_21_22_34_887913/caption_%d_6000_model.pth"%epoch))

    qid2caption_train = evaluate_caption(model, train_loader, caption_dictionary.idx2word, step = 1000000000)
    cPickle.dump(qid2caption_train, open('data/qid2caption_bs_%d_train.pkl' % epoch, 'wb'))
    qid2caption_val = evaluate_caption(model, eval_loader, caption_dictionary.idx2word, step = 1000000000)
    cPickle.dump(qid2caption_val, open('data/qid2caption_bs_%d_val.pkl'%epoch, 'wb'))



opt = opts.parse_opt()
train(opt)



