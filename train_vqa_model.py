from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from optim import WarmupOptimizer
import torch.optim as optim
import numpy as np
import time
import os
from six.moves import cPickle
import opts
import models
from misc.utils import *
from dataset import *
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model, dataloader, with_gt_caption=0):
    score = 0.0
    num_data = 0.0
    model.eval()
    score1 = 0.0
    score2 = 0.0
    for v, q, a, c, rc, vm, _ in iter(dataloader):
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()  # boxes not used
        c = c.long().cuda()
        rc = rc.long().cuda()
        pred, pred1, pred2 = model(v, q, c, rc, with_gt_caption, 'VQA')
        batch_score = compute_score_with_logits(pred, a.data).sum()
        batch_score1 = compute_score_with_logits(pred1, a.data).sum()
        batch_score2 = compute_score_with_logits(pred2, a.data).sum()
        score += batch_score
        score1 += batch_score1
        score2 += batch_score2
        num_data += pred.size(0)

    score /= num_data
    score1 /= num_data
    score2 /= num_data
    model.train()
    return score, score1, score2


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)


def train(opt):
    # Deal with feature things before anything

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')  # question dictionary
    caption_dictionary = Dictionary.load_from_file('data/caption_dictionary.pkl')

    train_dset = CaptionQAIMGDataset('train', dictionary, caption_dictionary, caption_dir = opt.caption_dir + '_train.pkl')
    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=0)
    eval_dset = CaptionQAIMGDataset('val', dictionary, caption_dictionary,caption_dir = opt.caption_dir + '_val.pkl')
    eval_loader = DataLoader(eval_dset,  opt.batch_size, shuffle=False, num_workers=0)
    opt.vocab_size = len(caption_dictionary.word2idx)
    opt.n_tokens = len(dictionary.word2idx)
    opt.seq_length = 17
    model = models.setup(opt).cuda()
    print("Printing Model")
    print(model)
    model.apply(weights_init_kn)

    # Assure in training mode
    
    model.train()
    vqa_optim = torch.optim.Adamax([{'params': model.w_emb.parameters()},
                                    {'params': model.q_emb.parameters()},
                                    {'params': model.classifier2.parameters()},
                                    {'params': model.v_att_1.parameters()},
                                    {'params': model.v_att_2.parameters()},
                                    {'params': model.v_net.parameters()},
                                    {'params': model.q_net.parameters()},
                                    {'params': model.classifier.parameters()},
                                    {'params': model.cw_emb.parameters()},
                                    {'params': model.cq_emb.parameters()},
                                    {'params': model.caption_w_emb.parameters()},
                                    {'params': model.caption_emb.parameters()},
                                    {'params': model.c_net.parameters()},
                                    {'params': model.classifier1.parameters()},
                                    {'params': model.c_att_1.parameters()},
                                    {'params': model.c_att_2.parameters()}, ],
                                   lr=opt.learning_rate, weight_decay=0)

    opt.checkpoint_path = 'saved_models/%s' % (
        str(datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_'))
    os.mkdir(opt.checkpoint_path)
    log_file = open(opt.checkpoint_path + '/log.txt', 'w')
    print(opt, file=log_file)
    log_file.flush()
    max_eval_score = 0
    dp_model = nn.DataParallel(model)
    decay_rate = [1] * 16

    for params_idx in range(len(vqa_optim.param_groups)):
        param_group = vqa_optim.param_groups[params_idx]
        print('CURRENT_LR: ', param_group['lr'], file=log_file)
        param_group['lr'] *= decay_rate[params_idx]

    count = 0
    for epoch in range(opt.train_vqa_epochs):
        i = 0
        losses = 0.0
        count += 1
        print("Epoch: %d/%d" %(count, opt.train_vqa_epochs))
        for v, q, a, c, rc, vm, _ in tqdm(iter(train_loader)):
            vqa_optim.zero_grad()
            v = v.cuda()
            q = q.cuda().long()
            a = a.cuda()
            c = c.long().cuda()
            rc = rc.long().cuda()
            pred, pred1, pred2 = dp_model(v, q, c, rc, 1, 'VQA')
            loss0 = instance_bce_with_logits(pred, a) # joint
            loss1 = instance_bce_with_logits(pred1, a) # caption
            loss2 = instance_bce_with_logits(pred2, a) # visual
            loss = loss0 * opt.joint_weight + loss1 * opt.caption_weight + loss2* opt.visual_weight
            loss.backward()
            losses += loss.item()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            vqa_optim.step()
            print('LOSS: ', loss.item())
            if (i + 1) % 1 == 0:
                print('LOSS: ', loss.item(), file=log_file)
                log_file.flush()
            i += 1
            if i % 2000 == 0:
                eval_score, eval_score1, eval_score2 = evaluate(dp_model, eval_loader, 1)
                print('VALIDATION: current_score', eval_score, 'best_val_score ', max_eval_score,\
                    'current_score1', eval_score1, 'current_score2 ', eval_score2, file=log_file)
                log_file.flush()
                torch.save(model.state_dict(), opt.checkpoint_path + '/vqa_model.pth')
                if eval_score > max_eval_score:
                    torch.save(model.state_dict(), 'vqa_models/vqa_model-best.pth')
                    max_eval_score = eval_score


opt = opts.parse_opt()
train(opt)






