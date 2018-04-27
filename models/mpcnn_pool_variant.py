import numpy as np
import torch
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNPoolVariant(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNPoolVariant, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_pool_variant'

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = len(self.pooling_funcs) * len(self.filter_widths) * COMP_2_COMPONENTS
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
            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent)
            if 'max' in self.pooling_funcs:
                block_a[ws]['max'] = F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)
            if 'min' in self.pooling_funcs:
                block_a[ws]['min'] = F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)
            if 'mean' in self.pooling_funcs:
                block_a[ws]['mean'] = F.avg_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)

            if ws not in block_b:
                block_b[ws] = {}
            per_dim_conv_out = self.per_dim_conv_layers[ws - 1](sent)
            if 'max' in self.pooling_funcs:
                block_b[ws]['max'] = F.max_pool1d(per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.n_word_dim, self.n_per_dim_filters)
            if 'min' in self.pooling_funcs:
                block_b[ws]['min'] = F.max_pool1d(-1 * per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            if 'mean' in self.pooling_funcs:
                block_b[ws]['mean'] = F.avg_pool1d(per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)

        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in self.pooling_funcs:
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                comparison_feats.append(F.cosine_similarity(x1, x2))
                comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.stack(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b, sent1_nonstatic=None, sent2_nonstatic=None):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in self.pooling_funcs:
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                        comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                        comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                        comparison_feats.append(torch.abs(x1 - x2))

        for pool in self.pooling_funcs:
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
