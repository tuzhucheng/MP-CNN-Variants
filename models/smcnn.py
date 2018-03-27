"""
An implementation of Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
https://dl.acm.org/citation.cfm?id=2767738
"""

import torch.nn as nn

from models.smcnn_variant_base import SMCNNVariantBase


class SMCNN(SMCNNVariantBase):

    def __init__(self, n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(SMCNN, self).__init__(n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'smcnn'
        self.n_word_dim = n_word_dim
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv

        self.in_channels = n_word_dim if attention == 'none' else 2*n_word_dim

        padding = filter_width - 1 if wide_conv else 0

        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels, n_filters, filter_width, padding=padding),
            nn.ReLU()
        )

        # compute number of inputs to first hidden layer
        n_feat = 2*n_filters + ext_feats

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )
