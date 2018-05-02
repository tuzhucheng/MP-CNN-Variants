import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNPoolVariant(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNPoolVariant, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_pool_variant'

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = len(self.pooling_funcs) * self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            len(self.pooling_funcs) * ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            len(self.pooling_funcs) * 3 +
            # comparison units from per-dim conv
            len(self.pooling_funcs) * (len(self.filter_widths) - 1) * self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _add_layers(self):
        super(MPCNNPoolVariant, self)._add_layers()
        per_dim_conv_layers_mean = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            padding = ws-1 if self.wide_conv else 0

            per_dim_conv_layers_mean.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

        self.per_dim_conv_layers_mean = nn.ModuleList(per_dim_conv_layers_mean)

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                if ws not in block_a:
                    block_a[ws] = {}
                if 'max' in self.pooling_funcs:
                    block_a[ws]['max'] = F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                if 'min' in self.pooling_funcs:
                    block_a[ws]['min'] = F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                if 'mean' in self.pooling_funcs:
                    block_a[ws]['mean'] = F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                continue

            if ws not in block_a:
                block_a[ws] = {}
            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            holistic_conv_out_min = self.holistic_conv_layers_min[ws - 1](sent)
            holistic_conv_out_mean = self.holistic_conv_layers_mean[ws - 1](sent)
            if 'max' in self.pooling_funcs:
                block_a[ws]['max'] = F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters)
            if 'min' in self.pooling_funcs:
                block_a[ws]['min'] = F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters)
            if 'mean' in self.pooling_funcs:
                block_a[ws]['mean'] = F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)

            if ws not in block_b:
                block_b[ws] = {}
            per_dim_conv_out_max = self.per_dim_conv_layers_max[ws - 1](sent)
            per_dim_conv_out_min = self.per_dim_conv_layers_min[ws - 1](sent)
            per_dim_conv_out_mean = self.per_dim_conv_layers_mean[ws - 1](sent)
            if 'max' in self.pooling_funcs:
                block_b[ws]['max'] = F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.n_word_dim, self.n_per_dim_filters)
            if 'min' in self.pooling_funcs:
                block_b[ws]['min'] = F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            if 'mean' in self.pooling_funcs:
                block_b[ws]['mean'] = F.avg_pool1d(per_dim_conv_out_mean, per_dim_conv_out_mean.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)

        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        return self._horizontal_comparison(sent1_block_a, sent2_block_a, pooling_types=self.pooling_funcs)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b, sent1_nonstatic=None, sent2_nonstatic=None):
        return self._vertical_comparison(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b, holistic_pooling_types=self.pooling_funcs,
                                            per_dim_pooling_types=self.pooling_funcs)
