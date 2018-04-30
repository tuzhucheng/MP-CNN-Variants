import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNSingleWindow(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        self.filter_width = filter_widths[-2]
        super(MPCNNSingleWindow, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_single_window'

    def _add_layers(self):
        padding = self.filter_width - 1 if self.wide_conv else 0

        self.holistic_conv_layer_max = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_holistic_filters, self.filter_width, padding=padding),
            nn.Tanh()
        )

        self.holistic_conv_layer_min = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_holistic_filters, self.filter_width, padding=padding),
            nn.Tanh()
        )

        self.holistic_conv_layer_mean = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_holistic_filters, self.filter_width, padding=padding),
            nn.Tanh()
        )

        self.per_dim_conv_layer_max = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, self.filter_width, padding=padding, groups=self.in_channels),
            nn.Tanh()
        )

        self.per_dim_conv_layer_min = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, self.filter_width, padding=padding, groups=self.in_channels),
            nn.Tanh()
        )

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = 3 * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from per-dim conv
            2 * self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}

        holistic_conv_out_max = self.holistic_conv_layer_max(sent)
        holistic_conv_out_min = self.holistic_conv_layer_min(sent)
        holistic_conv_out_mean = self.holistic_conv_layer_mean(sent)
        block_a[self.filter_width] = {
            'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters),
            'min': F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters),
            'mean': F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)
        }

        per_dim_conv_out_max = self.per_dim_conv_layer_max(sent)
        per_dim_conv_out_min = self.per_dim_conv_layer_min(sent)
        block_b[self.filter_width] = {
            'max': F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
            'min': F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
        }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            x1 = sent1_block_a[self.filter_width][pool]
            x2 = sent2_block_a[self.filter_width][pool]
            comparison_feats.append(F.cosine_similarity(x1, x2))
            comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.stack(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            x1 = sent1_block_a[self.filter_width][pool]
            x2 = sent2_block_a[self.filter_width][pool]
            comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
            comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
            comparison_feats.append(torch.abs(x1 - x2))

        for pool in ('max', 'min'):
            oG_1B = sent1_block_b[self.filter_width][pool]
            oG_2B = sent2_block_b[self.filter_width][pool]
            for i in range(0, self.n_per_dim_filters):
                x1 = oG_1B[:, :, i]
                x2 = oG_2B[:, :, i]
                comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)
