import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert = Bert(False, './temp', True)
hidden_size = 512
num_hidden_layers = 6
num_attention_heads = 8
intermediate_size = 512
hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
bert_config = BertConfig(bert.model.config.vocab_size,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_hidden_layers,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_probs_dropout_prob=attention_probs_dropout_prob)
bert.model = BertModel(bert_config)
copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
class BertEmbeddings(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()

