import torch.nn as nn

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

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = 3 * len(self.filter_widths) * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 3 +
            # comparison units from per-dim conv
            2 * self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats
