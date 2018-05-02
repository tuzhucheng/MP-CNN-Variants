import numpy as np
import torch
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNCompUnit2Only(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNCompUnit2Only, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_comp_unit2_only'

    def _get_n_feats(self):
        COMP_2_COMPONENTS = 2
        n_feats_h = 3 * self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * ((len(self.filter_widths) - 1) ** 2) * COMP_2_COMPONENTS +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 2 +
            # comparison units from per-dim conv
            2 * (len(self.filter_widths) - 1) * self.n_per_dim_filters * COMP_2_COMPONENTS
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        return self._horizontal_comparison(sent1_block_a, sent2_block_a, comparison_types=('cosine', 'euclidean'))

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        return self._vertical_comparison(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b,
                                         comparison_types=('cosine', 'euclidean'))
