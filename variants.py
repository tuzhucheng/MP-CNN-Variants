"""
Factory for returning different kinds of model variants.
"""
from models.mpcnn import MPCNN
from models.smcnn import SMCNN
from models.smcnn_multi_window import SMCNNMultiWindow
from models.mpcnn_pool_max_only import MPCNNPoolMaxOnly
from models.mpcnn_pool_mean_sym import MPCNNPoolMeanSymmetrical

import numpy as np


class VariantFactory(object):

    @staticmethod
    def get_model(args, dataset_cls):
        if args.arch == 'mpcnn':
            filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
            model = MPCNN(args.word_vectors_dim, args.holistic_filters, args.per_dim_filters, filter_widths,
                          args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, args.sparse_features,
                          args.attention)
        elif args.arch == 'mpcnn_pool_max_only':
            filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
            model = MPCNNPoolMaxOnly(args.word_vectors_dim, args.holistic_filters, args.per_dim_filters, filter_widths,
                          args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, args.sparse_features,
                          args.attention)
        elif args.arch == 'mpcnn_pool_mean_sym':
            filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
            model = MPCNNPoolMeanSymmetrical(args.word_vectors_dim, args.holistic_filters, args.per_dim_filters, filter_widths,
                          args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, args.sparse_features,
                          args.attention)
        elif args.arch == 'smcnn':
            model = SMCNN(args.word_vectors_dim, args.holistic_filters, args.max_window_size, args.hidden_units,
                          dataset_cls.NUM_CLASSES, args.dropout, args.sparse_features, args.attention)
        elif args.arch == 'smcnn_multi_window':
            filter_widths = list(range(1, args.max_window_size + 1))
            model = SMCNNMultiWindow(args.word_vectors_dim, args.holistic_filters, filter_widths, args.hidden_units,
                          dataset_cls.NUM_CLASSES, args.dropout, args.sparse_features)
        else:
            raise ValueError('Unrecognized model variant')

        return model
