from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import models
import cPickle
from tqdm import tqdm
from dataset import *
from datetime import datetime
from torch.utils.data import DataLoader
from misc.utils import *
from tqdm import tqdm
from datetime import datetime

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    time_now = str(datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '_' + time_now+'.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model_path, opt):
    dictionary = Dictionary.load_from_file('dictionary.pkl')  # question dictionary
    caption_dictionary = Dictionary.load_from_file('caption_dictionary.pkl')
    actual_batch_size = 32
    eval_dset = CaptionQAIMGDataset('val', dictionary, caption_dictionary)
    eval_loader = DataLoader(eval_dset, actual_batch_size, shuffle=False, num_workers=0)
    opt.vocab_size = len(caption_dictionary.word2idx)
    opt.n_tokens = len(dictionary.word2idx)
    opt.seq_length = 17  # loader.seq_length
    model = models.setup(opt)

    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # Make sure in the evaluation mode
    model.eval()
    predictions = []
    #train_statistics = {}
    for v, q, a, c, rc, qids in tqdm(iter(eval_loader)):
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        # forward the model to also get generated samples for each image
        v = v.cuda()
        q = q.cuda()
        with torch.no_grad():
            seq, _ = model._sample_beam(v, q, 0, 0)
            #seqp, _, _, hidden_states, _, _, correct, addition_output = model._forward(att_feats, labels, reverse_seqs, questions, answers, att_masks, state_weights = state_weights)
            #seq = seqp.max(2)[1]
            seq = seq.data
        # Print beam search
        sents = decode_sequence(cPickle.load(open('caption_dictionary.pkl'))[1], seq)


        for k in xrange(v.size(0)):
            for j in xrange(1):
                sent = sents[k+j]
                sent = sent.replace('<unk> is', 'it is').replace('<unk> are', 'there are').replace('<unk> ', '')
                entry = {'caption': sent, 'question_id': qids[k]}
                predictions.append(entry)
                print('image %s: %s' %(entry['question_id'], entry['caption']))

    cPickle.dump(predictions, open('predictions_beam.pkl', 'wb'))
    stat = language_eval('COCO', predictions, 'topdown', 'val')
    model.train()
    print (stat)
    
    return 0
