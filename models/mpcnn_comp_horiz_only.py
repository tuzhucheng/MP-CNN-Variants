import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn_variant_base import MPCNNVariantBase


class MPCNNCompHorizOnly(MPCNNVariantBase):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNCompHorizOnly, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_comp_horiz_only'
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv
        holistic_conv_layers = []
        per_dim_conv_layers = []

        self.in_channels = n_word_dim if attention == 'none' else 2*n_word_dim

        for ws in filter_widths:
            if np.isinf(ws):
                continue

            padding = ws - 1 if wide_conv else 0

            holistic_conv_layers.append(nn.Sequential(
                nn.Conv1d(self.in_channels, n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            per_dim_conv_layers.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * n_per_dim_filters, ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)
        self.per_dim_conv_layers = nn.ModuleList(per_dim_conv_layers)

        # compute number of inputs to first hidden layer
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + n_holistic_filters, 2 + self.in_channels, 2
        EXT_FEATS = 4 if ext_feats else 0
        n_feat_h = 3 * len(self.filter_widths) * COMP_2_COMPONENTS
        n_feat = n_feat_h + EXT_FEATS

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {
                    'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'min': F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'mean': F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'min': F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'mean': F.avg_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

            per_dim_conv_out = self.per_dim_conv_layers[ws - 1](sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                batch_size = x1.size()[0]
                comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.cat(comparison_feats, dim=1)

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        # Sentence modeling module
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        combined_feats = [feat_h, ext_feats] if self.ext_feats else [feat_h]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
