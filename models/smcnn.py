"""
An implementation of Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
https://dl.acm.org/citation.cfm?id=2767738
"""

import torch.nn as nn

from models.smcnn_variant_base import SMCNNVariantBase


class SMCNN(SMCNNVariantBase):

    def __init__(self, n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention):
        super(SMCNN, self).__init__(n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention)
        self.arch = 'smcnn'
        self.n_word_dim = n_word_dim
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.ext_feats = ext_feats
        self.attention = attention

        self.in_channels = n_word_dim if attention == 'none' else 2*n_word_dim

        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels, n_filters, filter_width),
            nn.ReLU()
        )

        # compute number of inputs to first hidden layer
        EXT_FEATS = 4 if ext_feats else 0
        n_feat = 2*n_filters + EXT_FEATS

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )
