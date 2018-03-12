"""
Factory for returning different kinds of model variants.
"""
from models.mpcnn import MPCNN
from models.smcnn import SMCNN
from models.smcnn_multi_window import SMCNNMultiWindow
from models.mpcnn_pool_max_only import MPCNNPoolMaxOnly
from models.mpcnn_pool_mean_sym import MPCNNPoolMeanSymmetrical
from models.mpcnn_pool_no_mean_sym import MPCNNPoolNoMeanSymmetrical
from models.mpcnn_comp_horiz_only import MPCNNCompCompHorizOnly

import numpy as np


class VariantFactory(object):

    @staticmethod
    def get_model(args, dataset_cls):
        if args.arch.startswith('mpcnn'):
            model_map = {
                'mpcnn': MPCNN,
                'mpcnn_pool_max_only': MPCNNPoolMaxOnly,
                'mpcnn_pool_mean_sym': MPCNNPoolMeanSymmetrical,
                'mpcnn_pool_no_mean_sym': MPCNNPoolNoMeanSymmetrical,
                'mpcnn_comp_horiz_only': MPCNNCompCompHorizOnly
            }

            filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
            model = model_map[args.arch](args.word_vectors_dim, args.holistic_filters, args.per_dim_filters, filter_widths,
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
