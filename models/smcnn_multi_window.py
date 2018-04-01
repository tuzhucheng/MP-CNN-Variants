"""
SM Model with multiple window sizes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.smcnn_variant_base import SMCNNVariantBase


class SMCNNMultiWindow(SMCNNVariantBase):

    def __init__(self, n_word_dim, n_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(SMCNNMultiWindow, self).__init__(n_word_dim, n_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'smcnn_multi_window'
        self.n_word_dim = n_word_dim
        self.n_filters = n_filters
        self.filter_widths = filter_widths
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv

        self.in_channels = n_word_dim if attention == 'none' else 2 * n_word_dim

        conv_layers = []
        for ws in filter_widths:
            padding = ws - 1 if wide_conv else 0
            conv_layers.append(nn.Sequential(
                nn.Conv1d(self.in_channels, n_filters, ws, padding=padding),
                nn.ReLU()
            ))

        self.conv_layers = nn.ModuleList(conv_layers)

        # compute number of inputs to first hidden layer
        n_feat = 2*n_filters*len(filter_widths) + ext_feats

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )

    def convolve(self, sent):
        combined = []
        for i, ws in enumerate(self.filter_widths):
            x = self.conv_layers[i](sent)
            x = F.max_pool1d(x, x.size(2))
            combined.append(x)

        return torch.cat(combined, dim=1)

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, sent1_nonstatic=None, sent2_nonstatic=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        xq = self.convolve(sent1)
        xd = self.convolve(sent2)

        combined_feats = [xq, xd, ext_feats] if self.ext_feats else [xq, xd]
        feat_all = torch.cat(combined_feats, dim=1)
        feat_all = feat_all.view(-1, feat_all.size(1))

        preds = self.final_layers(feat_all)
        return preds
