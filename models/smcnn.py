"""
An implementation of Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
https://dl.acm.org/citation.cfm?id=2767738
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMCNN(nn.Module):

    def __init__(self, n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention):
        super(SMCNN, self).__init__()
        self.arch = 'smcnn'
        self.n_word_dim = n_word_dim
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.ext_feats = ext_feats
        self.attention = attention

        self.in_channels = n_word_dim if not attention else 2*n_word_dim

        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels, n_filters, filter_width),
            nn.Tanh()
        )

        # compute number of inputs to first hidden layer
        EXT_FEATS = 4 if ext_feats else 0
        n_feat = n_filters + EXT_FEATS

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )

    def concat_attention(self, sent1, sent2):
        sent1_transposed = sent1.transpose(1, 2)
        attention_dot = torch.bmm(sent1_transposed, sent2)
        sent1_norms = torch.norm(sent1_transposed, p=2, dim=2, keepdim=True)
        sent2_norms = torch.norm(sent2, p=2, dim=1, keepdim=True)
        attention_norms = torch.bmm(sent1_norms, sent2_norms)
        attention_matrix = attention_dot / attention_norms

        attention_weight_vec1 = F.softmax(attention_matrix.sum(2), 1)
        attention_weight_vec2 = F.softmax(attention_matrix.sum(1), 1)
        attention_weighted_sent1 = attention_weight_vec1.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent1
        attention_weighted_sent2 = attention_weight_vec2.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent2
        attention_emb1 = torch.cat((attention_weighted_sent1, sent1), dim=1)
        attention_emb2 = torch.cat((attention_weighted_sent2, sent2), dim=1)
        return attention_emb1, attention_emb2

    def forward(self, sent1, sent2, ext_feats=None):
        # Attention
        if self.attention:
            sent1, sent2 = self.concat_attention(sent1, sent2)

        xq = self.conv(sent1)
        xq = F.max_pool1d(xq, xq.size(2))
        xd = self.conv(sent2)
        xd = F.max_pool1d(xd, xd.size(2))
        combined_feats = [xq, xd, ext_feats] if self.ext_feats else [xq, xd]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
