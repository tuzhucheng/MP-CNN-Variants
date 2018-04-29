import numpy as np
import torch
import torch.nn.functional as F

from models.mpcnn_holistic_only import MPCNNHolisticOnly


class MPCNNCompVertHolisticOnly(MPCNNHolisticOnly):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(MPCNNCompVertHolisticOnly, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_comp_vert_holistic_only'

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 3
        )
        n_feats = n_feats_v + self.ext_feats
        return n_feats

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                        comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                        comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                        comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, sent1_nonstatic=None, sent2_nonstatic=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        # Sentence modeling module
        sent1_block_a = self._get_blocks_for_sentence(sent1)
        sent2_block_a = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a)
        combined_feats = [feat_v, ext_feats] if self.ext_feats else [feat_v]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
