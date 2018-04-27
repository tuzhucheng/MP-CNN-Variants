import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mpcnn import MPCNN


class MPCNNLiteMultichannel(MPCNN):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv, nonstatic):
        self.nonstatic = nonstatic
        super(MPCNNLiteMultichannel, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, 'none', wide_conv)  # No attention
        self.arch = 'mpcnn_lite_multichannel'  # aka MP-CNN Lite

    def _add_layers(self):
        holistic_conv_layers = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            padding = (0, ws - 1) if self.wide_conv else 0
            channels = 2 if self.nonstatic else 1

            holistic_conv_layers.append(nn.Sequential(
                nn.Conv2d(channels, self.n_holistic_filters, (self.n_word_dim, ws), padding=padding),
                nn.Tanh()
            ))

        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2
        INF_COMP_FEATS = 4 if self.nonstatic else 3
        n_feats_h = len(self.filter_widths) * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for max pooling for non-infinite widths
            ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for max pooling for infinite widths
            INF_COMP_FEATS
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                block_a[ws] = {
                    'max': F.max_pool2d(sent, (sent.size(2), sent.size(3))).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool2d(holistic_conv_out, (1, holistic_conv_out.size(3))).contiguous().view(-1, self.n_holistic_filters)
            }

        return block_a

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', ):
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                comparison_feats.append(F.cosine_similarity(x1, x2))
                comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.stack(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', ):
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
        # Adapt to 2D Conv
        sent1 = torch.unsqueeze(sent1, dim=1)
        sent2 = torch.unsqueeze(sent2, dim=1)

        if self.nonstatic:
            sent1_nonstatic = torch.unsqueeze(sent1_nonstatic, dim=1)
            sent2_nonstatic = torch.unsqueeze(sent2_nonstatic, dim=1)
            sent1 = torch.cat([sent1, sent1_nonstatic], dim=1)
            sent2 = torch.cat([sent2, sent2_nonstatic], dim=1)

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
