from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
#from dataloader import *
#from dataloaderraw import *
#import eval_utils
import argparse
import torch



import torch.optim as optim

from misc.utils import *
from dataset import *
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# Input arguments and options
parser = argparse.ArgumentParser()
#=============
# Data input settings
parser.add_argument('--input_json', type=str, default='data/coco.json',
                    help='path to the json file containing additional info and vocab')
parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                    help='path to the directory containing the preprocessed att feats')
parser.add_argument('--input_box_dir', type=str, default='data/cocotalk_box',
                    help='path to the directory containing the boxes of att feats')
parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

# Model settings
parser.add_argument('--caption_model', type=str, default="topdown",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt')
parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
parser.add_argument('--model_type', type=str, default='baseline',
                    help='baseline or hAttn')
parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')


parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')

# feature manipulation
parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
parser.add_argument('--use_box', type=int, default=0,
                    help='If use box features')
parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

# Optimization: General
parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=384,
                    help='minibatch size')
parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

#Optimization: for the Language Model
parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
parser.add_argument('--learning_rate', type=float, default=2e-3,
                    help='learning rate')
parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')
parser.add_argument('--caption_dir', type=str, default='data/qid2caption',
                        help='Maximum scheduled sampling prob.')
parser.add_argument('--caption_model_path', type=str, default='saved_models/2019_11_19_18_03_47_079166/',
                        help='Model path of caption file')

parser.add_argument('--train_mode', type=int, default=1,
                        help='Maximum scheduled sampling prob.')
parser.add_argument('--train_vqa_epochs', type=int, default=14,
                        help='Maximum scheduled sampling prob.')

# Evaluation/Checkpointing
parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

# misc
parser.add_argument('--id', type=str, default='topdown',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')
parser.add_argument('--pretrain_vqa', type=int, default=30,
                        help='if true then use 80k, else use 110k')
parser.add_argument('--pretrain_caption', type=int, default=30,
                        help='if true then use 80k, else use 110k')

parser.add_argument('--load_model', type=str, default='None',
                        help='if true then use 80k, else use 110k')
# Reward
parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
parser.add_argument('--bleu_reward_weight', type=float, default=0,
                        help='The reward weight from bleu4')
parser.add_argument('--epoch', type=int, default=9,
                        help='The reward weight from bleu4')

parser.add_argument('--joint_weight', type=float, default=0.,
                        help='The reward weight from cider')
parser.add_argument('--caption_weight', type=float, default=0.,
                        help='The reward weight from bleu4')
parser.add_argument('--visual_weight', type=float, default=0.,
                        help='The reward weight from bleu4')
parser.add_argument('--lr1', type=float, default=0.25,
                        help='The reward weight from bleu4')
parser.add_argument('--lr2', type=float, default=0.25,
                        help='The reward weight from bleu4')
#==========

# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--verbose_beam', type=int, default=1, 
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, 
                help='if we need to calculate loss.')

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data
    labels = torch.max(labels, 1)[1].data
    return torch.sum(logits == labels)
    print("logits.shape: ", logits.shape)
    print("labels.shape: ", labels.shape)
    print("logits: ", logits)
    print("labels: ", labels)
    
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    values, indices = scores.max(1)
    return indices

def eval_split(opt):
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')  # question dictionary
    caption_dictionary = Dictionary.load_from_file('data/caption_dictionary.pkl')
    #actual_batch_size = 32
    eval_dset = CaptionQAIMGDataset('val', dictionary, caption_dictionary,caption_dir = opt.caption_dir + '_val.pkl')
    eval_loader = DataLoader(eval_dset,  opt.batch_size, shuffle=False, num_workers=0)
    opt.vocab_size = len(caption_dictionary.word2idx)
    opt.n_tokens = len(dictionary.word2idx)
    opt.seq_length = 17  # loader.seq_length
    model = models.setup(opt)

    print("Loading model from: ", opt.model)
    model.load_state_dict(torch.load(opt.model))
    model.cuda()

    # Make sure in the evaluation mode
    model.eval()
    predictions = []
    #train_statistics = {}
    correct = 0
    total = 0
    for v, q, a, c, rc, _, qids in tqdm(iter(eval_loader)):
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        # forward the model to also get generated samples for each image
        #print("=============a===========")
        #print("a.shape", a.shape)
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        c = c.long().cuda()
        rc = rc.long().cuda()
        '''
        with torch.no_grad():
            seq, _ = model._sample_beam(v, q, 0, 0)
            #seqp, _, _, hidden_states, _, _, correct, addition_output = model._forward(att_feats, labels, reverse_seqs, questions, answers, att_masks, state_weights = state_weights)
            #seq = seqp.max(2)[1]
            seq = seq.data
        # Print beam search
        sents = decode_sequence(cPickle.load(open('data/caption_dictionary.pkl', 'rb'))[1], seq)
        '''
        with torch.no_grad():
            pred, pred1, pred2 = model(v, q, c, rc, 0, 'VQA')
            batch_indices = compute_score_with_logits(pred, a.data)
            batch_indices1 = compute_score_with_logits(pred1, a.data)
            batch_indices2 = compute_score_with_logits(pred2, a.data)
            print("batch_score.shape: ", batch_indices.shape)
            correct += batch_indices.item()
            total += 384
            accuracy = correct*100/total
            print("Total: %d, || Accuracy: %.2f"%(total, accuracy))
            
            #print("batch: ", batch_score)
        '''
        for k in range(v.size(0)):
            for j in range(1):
                sent = sents[k+j]
                sent = sent.replace('<unk> is', 'it is').replace('<unk> are', 'there are').replace('<unk> ', '')
                #print(sent)
                entry = {'caption': sent, 'question_id': qids[k]}
                predictions.append(entry)
                #print('image %s: %s' %(entry['question_id'], entry['caption']))
        '''
    cPickle.dump(predictions, open('predictions_beam.pkl', 'wb'))
    #stat = language_eval('COCO', predictions, 'topdown', 'val')
    model.train()
    #print (stat)
    return 0


opt = parser.parse_args()
eval_split(opt)



