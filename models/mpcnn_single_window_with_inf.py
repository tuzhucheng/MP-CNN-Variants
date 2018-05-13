import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNSingleWindowWithInf(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        filter_widths = filter_widths[-2:]
        super(MPCNNSingleWindowWithInf, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_single_window_with_inf'

    def _add_layers(self):
        padding = self.filter_widths[0] - 1 if self.wide_conv else 0

        self.holistic_conv_layer_max = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_holistic_filters, self.filter_widths[0], padding=padding),
            nn.Tanh()
        )

        self.holistic_conv_layer_min = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_holistic_filters, self.filter_widths[0], padding=padding),
            nn.Tanh()
        )

        self.holistic_conv_layer_mean = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_holistic_filters, self.filter_widths[0], padding=padding),
            nn.Tanh()
        )

        self.per_dim_conv_layer_max = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, self.filter_widths[0], padding=padding, groups=self.in_channels),
            nn.Tanh()
        )

        self.per_dim_conv_layer_min = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, self.filter_widths[0], padding=padding, groups=self.in_channels),
            nn.Tanh()
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

            holistic_conv_out_max = self.holistic_conv_layer_max(sent)
            holistic_conv_out_min = self.holistic_conv_layer_min(sent)
            holistic_conv_out_mean = self.holistic_conv_layer_mean(sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'min': F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'mean': F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

            per_dim_conv_out_max = self.per_dim_conv_layer_max(sent)
            per_dim_conv_out_min = self.per_dim_conv_layer_min(sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            }
        return block_a, block_b
