import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn_no_per_dim_no_multi_pooling_no_horiz import MPCNNNoPerDimNoMultiPoolingNoHoriz


class MPCNNNoPerDimNoMultiPoolingNoHorizNoInf(MPCNNNoPerDimNoMultiPoolingNoHoriz):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        filter_widths = [w for w in filter_widths if not np.isinf(w)]
        super(MPCNNNoPerDimNoMultiPoolingNoHorizNoInf, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_no_per_dim_no_multi_pooling_no_horiz_no_inf'

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2
        n_feats_v = (
            # comparison units from holistic conv for max pooling for non-infinite widths
            (len(self.filter_widths) ** 2) * COMP_1_COMPONENTS_HOLISTIC
        )
        n_feats = n_feats_v + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        for ws in self.filter_widths:
            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

        return block_a
