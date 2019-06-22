import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH, get_norm
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm
class CaptionDecoderRNN(nn.Module):
    def __init__(self, in_dim, num_hid, v_dim, caption_w_emb, caption_decoder_class, rnn_type='LSTM'):
        """Module for question embedding
        """
        super(CaptionDecoderRNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type == 'LSTM' else nn.GRUCell
        
        self.rnn_v = rnn_cls(
            in_dim + num_hid + v_dim, num_hid)

        self.rnn_c = rnn_cls(
            num_hid + v_dim , num_hid)
        
        self.v_a = nn.Linear(v_dim, num_hid)
        self.h_a = nn.Linear(num_hid, num_hid)
        self.a = nn.Linear(num_hid, 1)
        self.v_input = nn.Linear(v_dim, in_dim)
        self.att_softmax = nn.Softmax(dim = 1)
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.rnn_type = rnn_type
        self.caption_w_emb = caption_w_emb
        self.classifier = caption_decoder_class

    def forward(self, caption ,features ):
        # in_dim = v_dim
        # caption: [batch , sequence 20 , in_dim ] captions
        # features: [batch , 36 , v_dim]
        caption = self.caption_w_emb(caption)
        feat_mean = torch.mean(features, 1)
        v_rnn_inp = self.v_input(feat_mean)
        x = torch.cat(( v_rnn_inp.unsqueeze(1) , caption), dim = 1)
        
        batch = x.size(0)
        attention = [] 
        
        hidden_v = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        memory_v = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        hidden_c = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        memory_c = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        
        outputs = []
        
        for i in range(20):
        
            input = torch.cat( (feat_mean, hidden_c, x[:,i,:]) , dim = 1)
            # [b , v_dim + num_hid + v_dim] 
            hidden_v, memory_v = self.rnn_v(input, (hidden_v, memory_v))
            # [b  , num_hid]
            va = self.v_a(features) #[b , 36 , 512]
            ha = self.h_a(hidden_v) #[b, 512]
            ha = ha.unsqueeze(1).repeat(1,100,1)
            att = self.a(torch.tanh(va + ha)) # [b,36,1]
            att = self.att_softmax(att) # [b,36,1]

            attention.append(att)

            feat_hat = torch.mean( features * att, 1 )
            
            input_c = torch.cat((feat_hat, hidden_v), 1 )
            hidden_c, memory_c = self.rnn_c(input_c, (hidden_c, memory_c))
            
            outputs.append(hidden_c)
            #[20, b*5, 512]
            
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0 ,1 )
        attention = torch.stack(attention)
        attention = torch.transpose(attention, 0 ,1)
        repr = self.classifier(outputs)
        return repr, attention

    def beam_search(self, features, beam_size):
        """Samples captions for given image features (Greedy search)."""
        # features : [batch, 36, 2048 ]
        batch_img = features.size(0)
        caption = torch.zeros(batch_img * beam_size, 20)
        caption_ = torch.zeros(batch_img * beam_size, 20)
        batch = caption.size(0)
        feat_mean = torch.mean(features, 1)
        v_rnn_inp = self.v_input(feat_mean)

        v_rnn_inp = v_rnn_inp.unsqueeze(1).repeat(1,beam_size, 1).view(-1, 300)
        # v_rnn_inp [batch*5 , 300]
        feat_mean = feat_mean.unsqueeze(1).repeat(1,beam_size, 1).view(-1, 2048)
        #feat_mean [batch * 5, 2048]
        features = features.unsqueeze(1).repeat(1,beam_size, 1, 1).view(batch, 36, 2048)
        hidden_v = Variable(torch.zeros(batch, self.num_hid)).cuda()
        memory_v = Variable(torch.zeros(batch, self.num_hid)).cuda()
        hidden_c = Variable(torch.zeros(batch, self.num_hid)).cuda()
        memory_c = Variable(torch.zeros(batch, self.num_hid)).cuda()
        hidden_v_ = Variable(torch.zeros(batch, self.num_hid)).cuda()
        memory_v_ = Variable(torch.zeros(batch, self.num_hid)).cuda()
        hidden_c_ = Variable(torch.zeros(batch, self.num_hid)).cuda()
        memory_c_ = Variable(torch.zeros(batch, self.num_hid)).cuda()
        attention = []
        sig = nn.Sigmoid()
        #log = nn.Log()
        log_prob = torch.zeros(batch)

        
        for i in xrange(20):
            if i == 0:
                cur_words_emb = v_rnn_inp
                #[batch*5, 300]
            else:
                cur_words_emb = self.caption_w_emb(Variable(caption[:,i - 1].contiguous().type(torch.LongTensor), volatile=True).cuda())
                # [batch*5, 300]

            input = torch.cat((feat_mean, hidden_c, cur_words_emb), dim=1)
            #input [batch* 5, 2048 + 512+ 300]
            hidden_v, memory_v = self.rnn_v(input, (hidden_v, memory_v))
                # [b  , num_hid]
            va = self.v_a(features)  # [b * 5, 36 , 512]
            ha = self.h_a(hidden_v)  # [b, 512]
            ha = ha.unsqueeze(1).repeat(1, 36, 1)
            att = self.a(torch.tanh(va + ha))  # [b*5,36,1]
            att = self.att_softmax(att)  # [b*5,36,1]
            feat_hat = torch.mean(features * att, 1)
            input_c = torch.cat((feat_hat, hidden_v), 1)
            hidden_c, memory_c = self.rnn_c(input_c, (hidden_c, memory_c))
            #[batch*5, 512]

            repr = self.classifier(hidden_c)
            # [batch * 5, 31200]
            cur_logprob =  torch.log(sig(repr))
            dst_logprob, indices = torch.sort(cur_logprob.data, dim = 1, descending = True)
            # [batch * 5, 31200]
            tmp_prob = log_prob.view(batch_img, beam_size, 1 ).repeat(1,1,beam_size)
            if i != 1:
                for b in xrange(batch_img):
                    for src in xrange(beam_size):
                        for dst in xrange(beam_size):
                            cur_beam = b*beam_size + src
                            tmp_prob[b, src, dst] =  tmp_prob[b, src, dst] + dst_logprob[cur_beam, dst]
                for b in xrange(batch_img):
                    beam_square = tmp_prob[b, :,:].view( -1 )
                    sorted_beam_square, beam_square_idx = torch.sort(beam_square, descending = True)
                    for s in xrange(beam_size):
                        idx_tmp = beam_square_idx[s]
                        src = idx_tmp / beam_size
                        dst = idx_tmp % beam_size
                        logprob_tmp = sorted_beam_square[s]
                        for j in xrange(i):
                            caption_[b * beam_size + s, j] = caption[b *beam_size + src, j]
                        caption_[b *beam_size + s, i] =  indices[b*beam_size + src, dst]
                        log_prob[b *beam_size + s] = logprob_tmp
                        hidden_v_[b * beam_size + s, :] = hidden_v[b * beam_size + src, :]
                        hidden_c_[b * beam_size + s, :] = hidden_c[b * beam_size + src, :]
                        memory_v_[b * beam_size + s, :] = memory_v[b * beam_size + src, :]
                        memory_c_[b * beam_size + s, :] = memory_c[b * beam_size + src, :]
                        #if caption_[b *beam_size + s, i]  == '3131' :
                        #    log_prob[b *beam_size + s] += 7777777
            
            else:
                for b in xrange(batch_img):
                    for src in xrange(beam_size):
                        cur_beam = b*beam_size + src
                        log_prob[cur_beam] = dst_logprob[b*beam_size , src]
                        hidden_v_[cur_beam, :] = hidden_v[b * beam_size, :]
                        hidden_c_[cur_beam, :] = hidden_c[b * beam_size, :]
                        memory_v_[cur_beam, :] = memory_v[b * beam_size, :]
                        memory_c_[cur_beam, :] = memory_c[b * beam_size, :]
                        caption_[cur_beam, 0] = caption[b * beam_size, 0]
                        caption_[cur_beam, 1] =  indices[b*beam_size, src]
            


            hidden_v = hidden_v_
            hidden_c = hidden_c_
            memory_v = memory_v_
            memory_c = memory_c_
            caption = caption_
        return caption


class CaptionQuestionRNN(nn.Module):
    def __init__(self, c_dim, num_hid, q_dim,  nlayers, bidirect, dropout,rnn_type='LSTM'):
        """Module for question embedding
        """
        super(CaptionQuestionRNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        
        self.rnn_att = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)


        self.rnn_c = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)
        
        self.q_emb_for_c = nn.Linear(q_dim, num_hid)
        self.att_logits = nn.Linear(num_hid, 1)

        self.Sig = nn.Sigmoid()
        self.c_dim = c_dim
        self.q_dim = q_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = int(bidirect) + 1
        self.rnn_type = rnn_type

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, q_emb , c_emb ):
        # 
        # q_emb : [batch , q_dim ] 
        # c_emb : [batch * 5, 14 , c_dim]
        
        
        batch = c_emb.size(0)
        n_c = c_emb.size(0) /  q_emb.size(0)
        att_hidden = self.init_hidden(batch)
        self.rnn_att.flatten_parameters()
        self.rnn_c.flatten_parameters()
        
        c_emb_output, c_emb_hidden = self.rnn_att( c_emb, att_hidden)
        # (batch * 5, 14, hid_dim)
        q_emb_for_c = self.q_emb_for_c(q_emb)
        q_emb_for_c = q_emb_for_c.unsqueeze(1).unsqueeze(1).repeat(1, n_c, 20, 1).view(batch, 20, self.num_hid )
        # (batch * 5, 14, hid_dim)
        att_joint_repr = q_emb_for_c * c_emb_output
        att_logits = self.att_logits(att_joint_repr)
        att = self.Sig(att_logits)

        atted_c_emb = c_emb * att
        c_hidden = self.init_hidden(batch)
        output, c_hidden = self.rnn_att( atted_c_emb, c_hidden)

        return output[:, -1]

class CaptionQuestionImageRNN(nn.Module):
    def __init__(self, c_dim, num_hid, q_dim,  nlayers, bidirect, dropout,rnn_type='LSTM',v_dim = 2048):
        """Module for question embedding
        """
        super(CaptionQuestionImageRNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        norm_layer = get_norm('weight')
        self.rnn_att = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)


        self.rnn_c = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)

        
        self.q_emb_for_c = FCNet([q_dim, num_hid], dropout= 0.2, norm= 'weight', act= 'LeakyReLU')
        self.att_logits = norm_layer(nn.Linear(num_hid, 1),  dim=None)

        self.v_emb_for_c = FCNet([v_dim, num_hid], dropout= 0.2, norm= 'weight', act= 'LeakyReLU')
        self.v_att_logits = norm_layer(nn.Linear(num_hid, 1),  dim=None)

        self.Sig = nn.Sigmoid()
        self.c_dim = c_dim
        self.q_dim = q_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = int(bidirect) + 1
        self.rnn_type = rnn_type
        self.v_dim = v_dim

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, q_emb , c_emb, v_emb ):
        # 
        # q_emb : [batch , q_dim ] 
        # c_emb : [batch * 5, 14 , c_dim]
        # v_emb : [batch , v_dim]
        
        batch = c_emb.size(0)
        n_c = c_emb.size(0) /  q_emb.size(0)
        length = c_emb.size(1)
        att_hidden = self.init_hidden(batch)
        self.rnn_att.flatten_parameters()
        self.rnn_c.flatten_parameters()
        
        c_emb_output, c_emb_hidden = self.rnn_att( c_emb, att_hidden)
        # (batch * 5, 14, hid_dim)
        q_emb_for_c = self.q_emb_for_c(q_emb)
        q_emb_for_c = q_emb_for_c.unsqueeze(1).unsqueeze(1).repeat(1, n_c , length, 1).view(batch, length, self.num_hid )
        # (batch * 5, 14, hid_dim)        

        v_emb_for_c = self.v_emb_for_c(v_emb)
        # (batch ,  hid_dim)

        v_emb_for_c = v_emb_for_c.unsqueeze(1).unsqueeze(1).repeat(1, n_c , length, 1).view(batch, length, self.num_hid )
        # (batch * 5, 20, hid_dim)

        att_joint_repr = q_emb_for_c * c_emb_output
        att_logits = self.att_logits(att_joint_repr)
        att = self.Sig(att_logits)

        att_joint_repr_v = v_emb_for_c * c_emb_output
        v_att_logits = self.v_att_logits(att_joint_repr_v)
        v_att = self.Sig(v_att_logits)

        atted_c_emb = c_emb * att* ( 1 + v_att)
        c_hidden = self.init_hidden(batch)
        output, c_hidden = self.rnn_att( atted_c_emb, c_hidden)

        return output[:, -1]

class CaptionQuestionImageRNN0(nn.Module):
    def __init__(self, c_dim, num_hid, q_dim,  nlayers, bidirect, dropout,rnn_type='LSTM',v_dim = 2048):
        """Module for question embedding
        """
        super(CaptionQuestionImageRNN0, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        norm_layer = get_norm('weight')
        self.rnn_att = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)


        self.rnn_c = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)

        self.v_emb_for_c = FCNet([v_dim, num_hid], dropout= 0.2, norm= 'weight', act= 'LeakyReLU')
        self.v_att_logits = norm_layer(nn.Linear(num_hid, 1),  dim=None)

        self.Sig = nn.Sigmoid()
        self.c_dim = c_dim
        self.q_dim = q_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = int(bidirect) + 1
        self.rnn_type = rnn_type
        self.v_dim = v_dim

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, q_emb , c_emb, v_emb ):
        # 
        # q_emb : [batch , q_dim ] 
        # c_emb : [batch * 5, 14 , c_dim]
        # v_emb : [batch , v_dim]
        
        batch = c_emb.size(0)
        n_c = c_emb.size(0) /  q_emb.size(0)
        att_hidden = self.init_hidden(batch)
        self.rnn_att.flatten_parameters()
        self.rnn_c.flatten_parameters()
        
        c_emb_output, c_emb_hidden = self.rnn_att( c_emb, att_hidden)
        # (batch * 5, 14, hid_dim)

        v_emb_for_c = self.v_emb_for_c(v_emb)
        # (batch ,  hid_dim)

        v_emb_for_c = v_emb_for_c.unsqueeze(1).unsqueeze(1).repeat(1, n_c , 17, 1).view(batch, 17, self.num_hid )
        # (batch * 5, 20, hid_dim)

        att_joint_repr_v = v_emb_for_c * c_emb_output
        v_att_logits = self.v_att_logits(att_joint_repr_v)
        v_att = self.Sig(v_att_logits)

        atted_c_emb = c_emb * v_att
        c_hidden = self.init_hidden(batch)
        output, c_hidden = self.rnn_att( atted_c_emb, c_hidden)

        return output[:, -1]
class CaptionQuestionImageRNN3(nn.Module):
    def __init__(self, c_dim, num_hid, q_dim,  nlayers, bidirect, dropout,rnn_type='LSTM',v_dim = 2048):
        """Module for question embedding
        """
        super(CaptionQuestionImageRNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        norm_layer = get_norm('weight')
        self.rnn_att = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)


        self.rnn_c = rnn_cls( c_dim , num_hid, nlayers, bidirectional=bidirect, dropout=dropout,batch_first=True)

        
        self.q_emb_for_c = FCNet([q_dim, num_hid], dropout= 0.2, norm= 'weight', act= 'LeakyReLU')
        self.att_logits = norm_layer(nn.Linear(num_hid, 1),  dim=None)

        self.v_emb_for_c = FCNet([v_dim, num_hid], dropout= 0.2, norm= 'weight', act= 'LeakyReLU')
        self.v_att_logits = norm_layer(nn.Linear(num_hid, 1),  dim=None)

        self.Sig = nn.Sigmoid()
        self.c_dim = c_dim
        self.q_dim = q_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = int(bidirect) + 1
        self.rnn_type = rnn_type
        self.v_dim = v_dim

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, q_emb , c_emb, v_emb ):
        # 
        # q_emb : [batch , q_dim ] 
        # c_emb : [batch * 5, 14 , c_dim]
        # v_emb : [batch , v_dim]
        
        batch = c_emb.size(0)
        n_c = c_emb.size(0) /  q_emb.size(0)
        att_hidden = self.init_hidden(batch)
        self.rnn_att.flatten_parameters()
        self.rnn_c.flatten_parameters()
        
        c_emb_output, c_emb_hidden = self.rnn_att( c_emb, att_hidden)
        # (batch * 5, 14, hid_dim)
        q_emb_for_c = self.q_emb_for_c(q_emb)
        q_emb_for_c = q_emb_for_c.unsqueeze(1).unsqueeze(1).repeat(1, n_c , 20, 1).view(batch, 20, self.num_hid )
        # (batch * 5, 14, hid_dim)        

        v_emb_for_c = self.v_emb_for_c(v_emb)
        # (batch ,  hid_dim)

        v_emb_for_c = v_emb_for_c.unsqueeze(1).unsqueeze(1).repeat(1, n_c , 20, 1).view(batch, 20, self.num_hid )
        # (batch * 5, 20, hid_dim)

        att_joint_repr = q_emb_for_c * c_emb_output
        att_logits = self.att_logits(att_joint_repr)
        att = self.Sig(att_logits)

        att_joint_repr_v = v_emb_for_c * c_emb_output
        v_att_logits = self.v_att_logits(att_joint_repr_v)
        v_att = self.Sig(v_att_logits)

        atted_c_emb = c_emb * (1 + att)* ( 1 + v_att)
        c_hidden = self.init_hidden(batch)
        output, c_hidden = self.rnn_att( atted_c_emb, c_hidden)

        return output[:, -1]


class QuestionCaptionDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(QuestionCaptionDecoderRNN, self).__init__()
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_size, vocab_size)
        #self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths = 20):
        """Decode image feature vectors and generates captions."""
        #features [b*5, v_dim]
        #captions [b*5, 20, hid_dim]
        batch = features.size(0)
        embeddings = captions
        #[b* 5,20,v_dim]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        #[b * 5,21,v_dim]
        self.lstm.flatten_parameters()
        hiddens, _ = self.lstm(embeddings[:,:-1,:])
        #hiddens = hiddens.view(batch,5,21,-1)
        
        outputs = hiddens #self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()


class CaptionRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN, self).__init__()
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_size, vocab_size)
        #self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths = 20):
        """Decode image feature vectors and generates captions."""
        #features [b*5, v_dim]
        #captions [b*5, 20, hid_dim]
        batch = features.size(0)
        embeddings = captions
        #[b* 5,20,v_dim]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        #[b * 5,21,v_dim]
        self.lstm.flatten_parameters()
        hiddens, _ = self.lstm(embeddings[:,:-1,:])
        #hiddens = hiddens.view(batch,5,21,-1)
        
        outputs = hiddens #self.linear(hiddens)
        return outputs

        
def build_model(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    
    caption_w_emb = WordEmbedding(dataset.caption_dictionary.ntoken, emb_dim=300, dropout=dropW)
    caption_decoder_class = SimpleClassifier( in_dim=num_hid, hid_dim=2 * num_hid, out_dim= dataset.caption_dictionary.ntoken, dropout=dropC, norm= norm, act= activation)
    return  CaptionDecoderRNN(300, num_hid, 2048, caption_w_emb, caption_decoder_class) 
