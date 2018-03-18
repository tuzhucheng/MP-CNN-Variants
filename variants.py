"""
Factory for returning different kinds of model variants.
"""
from models.mpcnn import MPCNN
from models.smcnn import SMCNN
from models.smcnn_multi_window import SMCNNMultiWindow
from models.mpcnn_holistic_only import MPCNNHolisticOnly
from models.mpcnn_pool_max_only import MPCNNPoolMaxOnly
from models.mpcnn_pool_mean_sym import MPCNNPoolMeanSymmetrical
from models.mpcnn_pool_no_mean_sym import MPCNNPoolNoMeanSymmetrical
from models.mpcnn_comp_horiz_only import MPCNNCompHorizOnly
from models.mpcnn_comp_vert_only import MPCNNCompVertOnly
from models.mpcnn_comp_unit1_only import MPCNNCompUnit1Only
from models.mpcnn_comp_unit2_only import MPCNNCompUnit2Only
from models.mpcnn_holistic_pool_max_only import MPCNNHolisticPoolMaxOnly
from models.mpcnn_single_window import MPCNNSingleWindow
from models.mpcnn_single_window_with_inf import MPCNNSingleWindowWithInf

import numpy as np


class VariantFactory(object):

    @staticmethod
    def get_model(args, dataset_cls):
        if args.arch.startswith('mpcnn'):
            model_map = {
                'mpcnn': MPCNN,
                'mpcnn_holistic_only': MPCNNHolisticOnly,
                'mpcnn_pool_max_only': MPCNNPoolMaxOnly,
                'mpcnn_pool_mean_sym': MPCNNPoolMeanSymmetrical,
                'mpcnn_pool_no_mean_sym': MPCNNPoolNoMeanSymmetrical,
                'mpcnn_comp_horiz_only': MPCNNCompHorizOnly,
                'mpcnn_comp_vert_only': MPCNNCompVertOnly,
                'mpcnn_comp_unit1_only': MPCNNCompUnit1Only,
                'mpcnn_comp_unit2_only': MPCNNCompUnit2Only,
                'mpcnn_holistic_pool_max_only': MPCNNHolisticPoolMaxOnly,
                'mpcnn_single_window': MPCNNSingleWindow,
                'mpcnn_single_window_with_inf': MPCNNSingleWindowWithInf
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
