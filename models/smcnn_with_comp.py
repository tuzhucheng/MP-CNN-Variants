import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMCNNWithComp(nn.Module):

    """
    SM model but with comparison. Note this uses Tanh for comparison with MP-CNN variants.
    """

    def __init__(self, n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention):
        super(SMCNNWithComp, self).__init__()

        self.arch = 'smcnn_with_comp'
        self.n_word_dim = n_word_dim
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.ext_feats = ext_feats
        self.attention = attention

        self.in_channels = n_word_dim if not attention else 2*n_word_dim

        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels, n_filters, filter_width),
            nn.ReLU()
        )

        # compute number of inputs to first hidden layer
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + n_filters, 2
        EXT_FEATS = 4 if ext_feats else 0
        n_feat_h = COMP_2_COMPONENTS
        n_feat_v = (
            # comparison units from holistic conv for max pooling for non-infinite widths
            COMP_1_COMPONENTS_HOLISTIC
        )
        n_feat = n_feat_h + n_feat_v + EXT_FEATS

        self.final_layers = nn.Sequential(
            nn.Linear(n_feat, hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_classes),
            nn.LogSoftmax(1)
        )

    def concat_attention(self, sent1, sent2):
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
        attention_emb1 = torch.cat((attention_weighted_sent1, sent1), dim=1)
        attention_emb2 = torch.cat((attention_weighted_sent2, sent2), dim=1)
        return attention_emb1, attention_emb2

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        holistic_conv_out = self.conv(sent)
        block_a[self.filter_width] = {
            'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_filters)
        }

        return block_a

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        x1 = sent1_block_a[self.filter_width]['max']
        x2 = sent2_block_a[self.filter_width]['max']
        batch_size = x1.size()[0]
        comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
        comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        x1 = sent1_block_a[self.filter_width]['max']
        batch_size = x1.size()[0]
        x2 = sent2_block_a[self.filter_width]['max']
        comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
        comparison_feats.append(F.pairwise_distance(x1, x2))
        comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def forward(self, sent1, sent2, ext_feats=None):
        # Attention
        if self.attention:
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