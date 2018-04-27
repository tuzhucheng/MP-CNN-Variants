from models.mpcnn_pool_variant import MPCNNPoolVariant


class MPCNNPoolMeanSymmetrical(MPCNNPoolVariant):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        self.pooling_funcs = ['max', 'min', 'mean']
        super(MPCNNPoolMeanSymmetrical, self).__init__(n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv)
        self.arch = 'mpcnn_pool_mean_sym'
