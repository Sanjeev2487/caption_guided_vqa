
# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from fc import *
from classifier import *
from language_model import *
from attention import *
from caption_model import *



from .CaptionModel import CaptionModel

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class TopDownModel(CaptionModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__()

        self.vocab_size = opt.vocab_size
        self.n_tokens = opt.n_tokens
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.num_layers = 2
        self.core = TopDownCore(opt)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        self.w_emb = WordEmbedding(self.n_tokens , emb_dim=300, dropout=0.4)
        self.q_emb = QuestionEmbedding( 300, 1280, nlayers=1, bidirect=False, dropout=0.2, rnn_type='GRU')
        self.w_emb.init_embedding('data/glove6b_init_300d.npy')

        self.cw_emb = WordEmbedding(self.n_tokens , emb_dim=300, dropout=0.4)
        self.cq_emb = QuestionEmbedding( 300, 1280, nlayers=1, bidirect=False, dropout=0.2, rnn_type='GRU')
        self.cw_emb.init_embedding('data/glove6b_init_300d.npy')


        self.caption_w_emb = WordEmbedding(self.vocab_size, emb_dim=300, dropout=0.4)
        self.caption_w_emb.init_embedding('data/glove6b_caption_init_300d.npy')
        self.caption_emb = CaptionQuestionImageRNN(c_dim=300, num_hid=640 , q_dim = 1280,  nlayers=1, bidirect=False,
                                                   dropout=0.2, rnn_type='GRU')
        self.c_net = FCNet([640, 1280], dropout=0.1, norm='weight', act='ReLU')

        self.v_att_1 = Att_3(self.att_feat_size, 1280, 1280, 'weight', 'LeakyReLU', dropout=0.2)
        self.v_att_2 = Att_3(self.att_feat_size, 1280, 1280, 'weight', 'LeakyReLU', dropout=0.2)
        self.c_att_1 = Att_3(self.att_feat_size, 640, 1280, 'weight', 'LeakyReLU', dropout=0.2)
        self.c_att_2 = Att_3(self.att_feat_size, 640, 1280, 'weight', 'LeakyReLU', dropout=0.2)

        self.q_net = FCNet([1280, 1280], dropout= 0.1, norm= 'weight', act= 'LeakyReLU')
        self.v_net = FCNet([self.att_feat_size, 1280], dropout= 0.1, norm= 'weight', act= 'LeakyReLU')
        self.classifier = SimpleClassifier( in_dim=1280, hid_dim= 2 * 1280, out_dim=3129, dropout=0.5, norm= 'weight', act= 'LeakyReLU')
        self.classifier1 = SimpleClassifier( in_dim=1280, hid_dim= 2 * 1280, out_dim=3129, dropout=0.5, norm= 'weight', act= 'LeakyReLU')
        self.classifier2 = SimpleClassifier( in_dim=1280, hid_dim= 2 * 1280, out_dim=3129, dropout=0.5, norm= 'weight', act= 'LeakyReLU')
        
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, v, q, c, rc, with_gt_caption = 1, task = 'VQA'):
        n_c = 5
        batch_size = v.size(0)
        w_emb = self.w_emb(q)  # get word embeddings
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        att_1 = self.v_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch, 1, v_dim]
        att = att_1 + att_2
        att_feats = att * v #+ v
        v_emb_ = (att_feats).sum(1)  # [batch, v_dim]

        if with_gt_caption:
            rc = rc.view(-1, rc.size(-1))
            caption_w_emb = self.caption_w_emb(rc)  # get word embeddings
        else:
            seqs, _ = self._sample( v, q, c, rc)
            seqs_cpu = seqs.data.cpu().numpy()
            new_seqs = np.zeros((5 * batch_size, 17))
            n_wods = (seqs_cpu > 0).sum(1)
            for i in range(batch_size):
                new_seqs[i,-n_wods[i]:] = seqs_cpu[i, :n_wods[i]]
            seqs = torch.from_numpy(new_seqs).cuda()
            caption_w_emb = self.caption_w_emb(seqs.long())  # get word embeddings


        cw_emb = self.cw_emb(q)  # get word embeddings
        cq_emb = self.cq_emb(cw_emb)  # run GRU on word embeddings [batch, q_dim]
        c_emb = self.caption_emb(cq_emb, caption_w_emb, v_emb_)  # run GRU on word embeddings [batch, q_dim]

        att_3 = self.c_att_1(v.unsqueeze(1).repeat(1,n_c,1, 1).view(c_emb.size(0), 36, -1), c_emb)  # [batch, 1, v_dim]
        att_4 = self.c_att_2(v.unsqueeze(1).repeat(1,n_c,1, 1).view(c_emb.size(0), 36, -1), c_emb)  # [batch, 1, v_dim]
        att_3 = att_3.view(cq_emb.size(0), n_c, 36, -1).max(1)[0]
        att_4 = att_4.view(cq_emb.size(0), n_c, 36, -1).max(1)[0]

        # att = (att_1 + att_2)*(1 + att_3 + att_4)
        v_emb = ((att_feats)*(1 + att_3 + att_4)).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        c_repr = self.c_net(c_emb)
        c_repr, _ = c_repr.view(q_repr.size(0), 5, -1).max(1)

        joint_repr1 = q_repr * c_repr
        joint_repr2 = q_repr * v_repr
        joint_repr = joint_repr1 + joint_repr2
        pred = self.classifier(joint_repr)
        pred1 = self.classifier1(joint_repr1)
        pred2 = self.classifier2(joint_repr2)

        if task == 'VQA':
            return pred, pred1, pred2
        
        state = self.init_hidden(5 * batch_size)
        outputs = v.new_zeros(5 * batch_size, c.size(2) - 1, self.vocab_size + 1)
        seq = c.view(-1, c.size(-1))
        if 'caption' in task:
            # att_feats b 36 2048
            repeated_att_feats = att_feats.unsqueeze(1).repeat(1, 5, 1, 1).view(batch_size * 5, 36, -1)
            p_fc_feats, p_att_feats, pp_att_feats, _ = self._prepare_feature(repeated_att_feats.mean(1), repeated_att_feats, None)

            for i in range(c.size(-1) - 1):
                if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i].data.clone()
                        prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i].clone()
                # break if all the sequences end
                if i >= 1 and seq[:, i].sum() == 0:
                    break

                output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, None, state)
                outputs[:, i] = output

        return pred, outputs, repeated_att_feats, att_feats

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample(self, v, q, c, rc):
        n_c = 5
        batch_size = v.size(0)
        state = self.init_hidden(5 * batch_size)
        sample_max = 0
        decoding_constraint = 1
        temperature = 1.0
        w_emb = self.w_emb(q)  # get word embeddings
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        att_1 = self.v_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch, 1, v_dim]
        att = att_1 + att_2
        att_feats = att * v #+ v

        repeated_att_feats = att_feats.unsqueeze(1).repeat(1, 5, 1, 1).view(batch_size * 5, 36, -1)
        p_fc_feats, p_att_feats, pp_att_feats, _ = self._prepare_feature(repeated_att_feats.mean(1), repeated_att_feats, None)


        seq = v.new_zeros((5 * batch_size, self.seq_length), dtype=torch.long) + self.vocab_size
        seqLogprobs = v.new_zeros(5 * batch_size, self.seq_length)
        
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = p_fc_feats.new_zeros(5 * batch_size, dtype=torch.long)  + self.vocab_size

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, None, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp
            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break

            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _sample_beam(self, v, q, c, rc):
        beam_size = 3
        n_c = 5
        batch_size = v.size(0)
        sample_max = 0
        decoding_constraint = 1
        temperature = 1.0
        w_emb = self.w_emb(q)  # get word embeddings
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        att_1 = self.v_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch, 1, v_dim]
        att = att_1 + att_2
        att_feats = att * v #+ v

        p_fc_feats, p_att_feats, pp_att_feats, _ = self._prepare_feature(att_feats.mean(1), att_feats,  None)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = None #p_att_masks[5*k:5*k + 1].expand(*((beam_size,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = v.new_zeros([beam_size], dtype=torch.long) + self.vocab_size

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                          tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  tmp_att_masks, opt={'vocab_size' : self.vocab_size})
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), self.done_beams


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, state
