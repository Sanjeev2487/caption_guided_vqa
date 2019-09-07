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
        captions.append(caption)
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

def evaluate_caption(model, dataloader, idx2word, step = 10000):
    score = 0
    num_data = 0
    model.eval()
    captions = []
    i = 0
    for v, q, a, c, rc, vm,  _ in iter(dataloader):
        v = v.cuda()
        q = q.cuda()
        a = a.cuda() # boxes not used
        c = c.long().cuda()
        rc = rc.long().cuda()
        seq, seqLogprobs = model._sample(v, q, c, rc)
        captions.extend(decode_caption(idx2word, seq))
        if i == step:
            break
        else:
            i += 1
    model.train()
    return captions

def train(opt):
    # Deal with feature things before anything
    crit = LanguageModelCriterion()
    dictionary = Dictionary.load_from_file('data/dictionary.pkl') # question dictionary
    caption_dictionary = Dictionary.load_from_file('data/caption_dictionary.pkl')
    train_dset = CaptionQAIMGDataset('train', dictionary, caption_dictionary)
    actual_batch_size = 64
    train_loader = DataLoader(train_dset, actual_batch_size, shuffle=True, num_workers=0)
    eval_dset = CaptionQAIMGDataset('val', dictionary, caption_dictionary)
    eval_loader  = DataLoader(eval_dset, actual_batch_size, shuffle=False, num_workers=0)
    opt.vocab_size = len(caption_dictionary.word2idx)
    opt.n_tokens = len(dictionary.word2idx)
    opt.seq_length = 17 
    model = models.setup(opt).cuda()
    model.apply(weights_init_kn)

    update_lr_flag = True
    # Assure in training mode
    model.train()
    vqa_optim = torch.optim.Adamax([{'params': model.w_emb.parameters()},
                                    {'params': model.cw_emb.parameters()},
                                    {'params': model.cq_emb.parameters()},
                                    {'params': model.caption_w_emb.parameters()},
                                    {'params': model.caption_emb.parameters()},
                                    {'params': model.c_net.parameters()},
                                    {'params': model.q_emb.parameters()},
                                    {'params': model.classifier.parameters()},
                                    {'params': model.classifier1.parameters()},
                                    {'params': model.classifier2.parameters()},
                                    {'params': model.v_att_1.parameters()},
                                    {'params': model.v_att_2.parameters()},
                                    {'params': model.v_net.parameters()},
                                    {'params': model.c_att_1.parameters()},
                                    {'params': model.c_att_2.parameters()},
                                    {'params': model.q_net.parameters()}, ], lr=0.002, weight_decay=0)

    caption_optim = torch.optim.Adam([{'params' : model.embed.parameters()},
                                {'params': model.fc_embed.parameters()},
                                {'params': model.att_embed.parameters()},
                                {'params': model.logit.parameters()},
                                {'params': model.ctx2att.parameters()},
                                {'params': model.core.parameters()},] , lr=0.0005, weight_decay=0)

    opt.checkpoint_path = 'saved_models/%s' % (
        str(datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_'))
    os.mkdir(opt.checkpoint_path)
    log_file = open(opt.checkpoint_path + '/log.txt', 'w')
    print(opt, file=log_file)
    log_file.flush()
    max_eval_score = 0

    model.load_state_dict( torch.load("vqa_models/vqa_model-best.pth"))

    for epoch in xrange(opt.pretrain_caption):
        i = 0
        if update_lr_flag:
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            set_lr(caption_optim, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False
                
        for v, q, a, c, rc, vm,  _ in tqdm(iter(train_loader)):
            #print len(enumerate(train_loader))
            v = v.cuda()
            q = q.cuda().long()
            a = a.cuda() # boxes not used
            c = c.long().cuda()
            rc = rc.long().cuda()

            caption_optim.zero_grad()
            losses = 0.0

            pred, caption, repeated_att_feats, att_feats = model(v, q, c, rc, 1, 'caption')

            loss_vqa = instance_bce_with_logits(pred, a)

            c = c.view(-1, c.size(-1))
            nonzeros = (c < len(caption_dictionary)).sum(1)
            masks = np.zeros(c.shape)
            for b in xrange(c.shape[0]):
                masks[b, :nonzeros[b] + 2] = 1
            masks = torch.from_numpy(masks).cuda().float()

            loss_caption = crit(caption, c[:, 1:], masks[:, 1:])
            caption_grad = (torch.autograd.grad(loss_caption.mean(), repeated_att_feats, retain_graph=True)[0]).view(v.size(0), 5, 36, 2048)  # [b * 5, 36, 2048]
            vqa_grad = (torch.autograd.grad(loss_vqa, att_feats, retain_graph=True)[0]).view(v.size(0), 1, 36,  2048)  # [b, 1, 36, 2048]
            grads = (caption_grad * vqa_grad).sum(2).sum(2)  # b, 5
            caption_masks = (grads == grads.max(1)[0].unsqueeze(1)).float().view(-1, 1)
            caption_optim.zero_grad()
            loss = (((loss_caption * caption_masks).mean() * 5) )
            loss.backward()
            losses += loss.item()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            caption_optim.step()
                
            if (i+1) % (100) == 0:
                print('LOSS: ', losses, file=log_file)
                captions = evaluate_caption(model, eval_loader, caption_dictionary.idx2word, step = 10)
                print(captions)
                log_file.flush()
            i += 1
            if i % (2000) == 0:
                torch.save(model.state_dict(), opt.checkpoint_path + '/caption_%d_%d_model.pth'%(epoch, i))

        torch.save(model.state_dict(), opt.checkpoint_path + '/caption_model.pth')

opt = opts.parse_opt()
train(opt)



