import numpy as np
import torch
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNCompUnit1Only(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNCompUnit1Only, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_comp_unit1_only'

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM = 2 + self.n_holistic_filters, 2 + self.in_channels
        n_feats_h = 3 * self.n_holistic_filters * COMP_1_COMPONENTS_HOLISTIC
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

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        return self._horizontal_comparison(sent1_block_a, sent2_block_a, comparison_types=('cosine', 'euclidean', 'abs'))

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                        comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                        comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                        comparison_feats.append(torch.abs(x1 - x2))

        for pool in ('max', 'min'):
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, i]
                    x2 = oG_2B[:, :, i]
                    comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                    comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                    comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)
