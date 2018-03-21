import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn_variant_base import MPCNNVariantBase


class MPCNNSingleWindow(MPCNNVariantBase):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention):
        super(MPCNNSingleWindow, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention)

        filter_width = filter_widths[-2]

        self.arch = 'mpcnn_single_window'
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_width = filter_width
        self.ext_feats = ext_feats
        self.attention = attention

        self.in_channels = n_word_dim if attention == 'none' else 2*n_word_dim

        self.holistic_conv_layer = nn.Sequential(
            nn.Conv1d(self.in_channels, n_holistic_filters, filter_width),
            nn.Tanh()
        )

        self.per_dim_conv_layer = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels * n_per_dim_filters, filter_width, groups=self.in_channels),
            nn.Tanh()
        )

        # compute number of inputs to first hidden layer
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + n_holistic_filters, 2 + self.in_channels, 2
        EXT_FEATS = 4 if ext_feats else 0
        n_feat_h = 3 * COMP_2_COMPONENTS
        n_feat_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from per-dim conv
            2 * n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feat = n_feat_h + n_feat_v + EXT_FEATS

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

        holistic_conv_out = self.holistic_conv_layer(sent)
        block_a[self.filter_width] = {
            'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters),
            'min': F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters),
            'mean': F.avg_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)
        }

        per_dim_conv_out = self.per_dim_conv_layer(sent)
        block_b[self.filter_width] = {
            'max': F.max_pool1d(per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
            'min': F.max_pool1d(-1 * per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
        }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            x1 = sent1_block_a[self.filter_width][pool]
            x2 = sent2_block_a[self.filter_width][pool]
            batch_size = x1.size()[0]
            comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
            comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            x1 = sent1_block_a[self.filter_width][pool]
            batch_size = x1.size()[0]
            x2 = sent2_block_a[self.filter_width][pool]
            comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
            comparison_feats.append(F.pairwise_distance(x1, x2))
            comparison_feats.append(torch.abs(x1 - x2))

        for pool in ('max', 'min'):
            oG_1B = sent1_block_b[self.filter_width][pool]
            oG_2B = sent2_block_b[self.filter_width][pool]
            for i in range(0, self.n_per_dim_filters):
                x1 = oG_1B[:, :, i]
                x2 = oG_2B[:, :, i]
                batch_size = x1.size()[0]
                comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                comparison_feats.append(F.pairwise_distance(x1, x2))
                comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)
