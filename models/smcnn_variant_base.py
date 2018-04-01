import torch
import torch.nn as nn
import torch.nn.functional as F


class SMCNNVariantBase(nn.Module):

    def __init__(self, n_word_dim, n_filters, filter_width, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super(SMCNNVariantBase, self).__init__()

    def concat_attention(self, sent1, sent2, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        sent1_transposed = sent1.transpose(1, 2)
        attention_dot = torch.bmm(sent1_transposed, sent2)
        sent1_norms = torch.norm(sent1_transposed, p=2, dim=2, keepdim=True)
        sent2_norms = torch.norm(sent2, p=2, dim=1, keepdim=True)
        attention_norms = torch.bmm(sent1_norms, sent2_norms)
        attention_matrix = attention_dot / attention_norms

        attention_weight_vec1 = F.softmax(attention_matrix.sum(2), 1)
        attention_weight_vec2 = F.softmax(attention_matrix.sum(1), 1)

        if self.attention == 'idf' and word_to_doc_count is not None:
            for i, sent in enumerate(raw_sent1):
                for j, word in enumerate(sent.split(' ')):
                    attention_weight_vec1[i, j] /= word_to_doc_count.get(word, 1)

            for i, sent in enumerate(raw_sent2):
                for j, word in enumerate(sent.split(' ')):
                    attention_weight_vec2[i, j] /= word_to_doc_count.get(word, 1)

        attention_weighted_sent1 = attention_weight_vec1.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent1
        attention_weighted_sent2 = attention_weight_vec2.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent2
        attention_emb1 = torch.cat((attention_weighted_sent1, sent1), dim=1)
        attention_emb2 = torch.cat((attention_weighted_sent2, sent2), dim=1)
        return attention_emb1, attention_emb2

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, sent1_nonstatic=None, sent2_nonstatic=None):
        # Attention
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)

        xq = self.conv(sent1)
        xq = F.max_pool1d(xq, xq.size(2))
        xd = self.conv(sent2)
        xd = F.max_pool1d(xd, xd.size(2))
        combined_feats = [xq, xd, ext_feats] if self.ext_feats else [xq, xd]
        feat_all = torch.cat(combined_feats, dim=1)
        feat_all = feat_all.view(-1, feat_all.size(1))

        preds = self.final_layers(feat_all)
        return preds
