import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn_lite_multichannel import MPCNNLiteMultichannel


class MPCNNLiteMultiChannelAttention(MPCNNLiteMultichannel):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        # TODO Currently this does not support non-static embedding
        super(MPCNNLiteMultiChannelAttention, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv, False)
        self.arch = 'mpcnn_lite_multichannel_attention'

    def _add_layers(self):
        holistic_conv_layers = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            holistic_conv_layers.append(nn.Sequential(
                nn.Conv2d(2, self.n_holistic_filters, (self.n_word_dim, ws)),
                nn.Tanh()
            ))

        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2
        n_feats_h = self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for max pooling for non-infinite widths
            ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for max pooling for infinite widths
            4
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def concat_attention(self, sent1, sent2, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        if self.attention == 'modified_euclidean':
            attention_matrix = sent1.new_ones(sent1.size(0), sent1.size(2), sent2.size(2))
            # non-vectorized
            for b in range(sent1.size(0)):
                for i in range(sent1.size(2)):
                    for j in range(sent2.size(2)):
                        euclidean_dist = torch.sqrt(torch.sum((sent1[b, :, i] - sent2[b, :, j]) ** 2))
                        attention_matrix[b, i, j] = 1 / (1 + euclidean_dist)
        else:
            sent1_transposed = sent1.transpose(1, 2)
            attention_dot = torch.bmm(sent1_transposed, sent2)
            sent1_norms = torch.norm(sent1_transposed, p=2, dim=2, keepdim=True)
            sent2_norms = torch.norm(sent2, p=2, dim=1, keepdim=True)
            attention_norms = torch.bmm(sent1_norms, sent2_norms)
            attention_matrix = attention_dot / attention_norms

        attention_weight_vec1 = F.softmax(attention_matrix.sum(2), 1)
        attention_weight_vec2 = F.softmax(attention_matrix.sum(1), 1)

        attention_weighted_sent1 = attention_weight_vec1.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent1
        attention_weighted_sent2 = attention_weight_vec2.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent2

        sent1_4d = torch.unsqueeze(sent1, dim=1)
        sent2_4d = torch.unsqueeze(sent2, dim=1)
        attention_weighted_sent1_4d = torch.unsqueeze(attention_weighted_sent1, dim=1)
        attention_weighted_sent2_4d = torch.unsqueeze(attention_weighted_sent2, dim=1)

        attention_emb1 = torch.cat((sent1_4d, attention_weighted_sent1_4d), dim=1)
        attention_emb2 = torch.cat((sent2_4d, attention_weighted_sent2_4d), dim=1)
        return attention_emb1, attention_emb2

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, sent1_nonstatic=None, sent2_nonstatic=None):
        # Add attention feature map
        sent1, sent2 = self.concat_attention(sent1, sent2)

        # Sentence modeling module
        sent1_block_a = self._get_blocks_for_sentence(sent1)
        sent2_block_a = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a)
        combined_feats = [feat_h, feat_v, ext_feats] if self.ext_feats else [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
