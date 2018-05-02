import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn_variant_base import MPCNNVariantBase


class MPCNN(MPCNNVariantBase):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNN, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn'
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv

        self.in_channels = n_word_dim if attention == 'none' else 2*n_word_dim

        self._add_layers()

        # compute number of inputs to first hidden layer
        n_feats = self._get_n_feats()

        self.final_layers = nn.Sequential(
            nn.Linear(n_feats, hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )

    def _add_layers(self):
        holistic_conv_layers_max = []
        holistic_conv_layers_min = []
        holistic_conv_layers_mean = []
        per_dim_conv_layers_max = []
        per_dim_conv_layers_min = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            padding = ws-1 if self.wide_conv else 0

            holistic_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            holistic_conv_layers_min.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            holistic_conv_layers_mean.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

            per_dim_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

            per_dim_conv_layers_min.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

        self.holistic_conv_layers_max = nn.ModuleList(holistic_conv_layers_max)
        self.holistic_conv_layers_min = nn.ModuleList(holistic_conv_layers_min)
        self.holistic_conv_layers_mean = nn.ModuleList(holistic_conv_layers_mean)
        self.per_dim_conv_layers_max = nn.ModuleList(per_dim_conv_layers_max)
        self.per_dim_conv_layers_min = nn.ModuleList(per_dim_conv_layers_min)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = 3 * self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 3 +
            # comparison units from per-dim conv
            2 * (len(self.filter_widths) - 1) * self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

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

            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            holistic_conv_out_min = self.holistic_conv_layers_min[ws - 1](sent)
            holistic_conv_out_mean = self.holistic_conv_layers_mean[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'min': F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'mean': F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

            per_dim_conv_out_max = self.per_dim_conv_layers_max[ws - 1](sent)
            per_dim_conv_out_min = self.per_dim_conv_layers_min[ws - 1](sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        return self._horizontal_comparison(sent1_block_a, sent2_block_a)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        return self._vertical_comparison(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, sent1_nonstatic=None, sent2_nonstatic=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        # Sentence modeling module
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        combined_feats = [feat_h, feat_v, ext_feats] if self.ext_feats else [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
