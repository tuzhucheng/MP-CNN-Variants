"""
Factory for returning different kinds of model variants.
"""
from models.mpcnn import MPCNN
from models.smcnn import SMCNN
from models.smcnn_multi_window import SMCNNMultiWindow
from models.smcnn_with_comp import SMCNNWithComp
from models.mpcnn_holistic_only import MPCNNHolisticOnly
from models.mpcnn_pool_max_only import MPCNNPoolMaxOnly
from models.mpcnn_pool_mean_sym import MPCNNPoolMeanSymmetrical
from models.mpcnn_pool_no_mean_sym import MPCNNPoolNoMeanSymmetrical
from models.mpcnn_comp_horiz_only import MPCNNCompHorizOnly
from models.mpcnn_comp_horiz_abs_only_vert_u1 import MPCNNCompHorizAbsOnlyVertU1
from models.mpcnn_comp_vert_only import MPCNNCompVertOnly
from models.mpcnn_comp_vert_holistic_only import MPCNNCompVertHolisticOnly
from models.mpcnn_comp_unit1_only import MPCNNCompUnit1Only
from models.mpcnn_comp_unit2_only import MPCNNCompUnit2Only
from models.mpcnn_comp_abs_diff import MPCNNCompAbsDiff
from models.mpcnn_comp_cosine import MPCNNCompCosine
from models.mpcnn_comp_euclidean import MPCNNCompEuclidean
from models.mpcnn_holistic_pool_max_only import MPCNNHolisticPoolMaxOnly
from models.mpcnn_shared_filters import MPCNNSharedFilters
from models.mpcnn_no_inf import MPCNNNoInf
from models.mpcnn_single_window import MPCNNSingleWindow
from models.mpcnn_single_window_with_inf import MPCNNSingleWindowWithInf
from models.mpcnn_no_per_dim_no_multi_pooling import MPCNNNoPerDimNoMultiPooling
from models.mpcnn_no_per_dim_no_multi_pooling_no_horiz import MPCNNNoPerDimNoMultiPoolingNoHoriz
from models.mpcnn_no_per_dim_no_multi_pooling_no_horiz_no_inf import MPCNNNoPerDimNoMultiPoolingNoHorizNoInf
from models.mpcnn_lite_multichannel import MPCNNLiteMultichannel
from models.mpcnn_lite_multichannel_attention import MPCNNLiteMultiChannelAttention

import numpy as np


class VariantFactory(object):

    @staticmethod
    def get_model(args, dataset_cls):
        ext_feats = dataset_cls.EXT_FEATS if args.sparse_features else 0
        if args.arch.startswith('mpcnn'):
            model_map = {
                'mpcnn': MPCNN,
                'mpcnn_holistic_only': MPCNNHolisticOnly,
                'mpcnn_pool_max_only': MPCNNPoolMaxOnly,
                'mpcnn_pool_mean_sym': MPCNNPoolMeanSymmetrical,
                'mpcnn_pool_no_mean_sym': MPCNNPoolNoMeanSymmetrical,
                'mpcnn_comp_horiz_only': MPCNNCompHorizOnly,
                'mpcnn_comp_horiz_abs_only_vert_u1': MPCNNCompHorizAbsOnlyVertU1,
                'mpcnn_comp_vert_only': MPCNNCompVertOnly,
                'mpcnn_comp_vert_holistic_only': MPCNNCompVertHolisticOnly,
                'mpcnn_comp_unit1_only': MPCNNCompUnit1Only,
                'mpcnn_comp_unit2_only': MPCNNCompUnit2Only,
                'mpcnn_comp_abs_diff': MPCNNCompAbsDiff,
                'mpcnn_comp_cosine': MPCNNCompCosine,
                'mpcnn_comp_euclidean': MPCNNCompEuclidean,
                'mpcnn_holistic_pool_max_only': MPCNNHolisticPoolMaxOnly,
                'mpcnn_shared_filters': MPCNNSharedFilters,
                'mpcnn_no_inf': MPCNNNoInf,
                'mpcnn_single_window': MPCNNSingleWindow,
                'mpcnn_single_window_with_inf': MPCNNSingleWindowWithInf,
                'mpcnn_no_per_dim_no_multi_pooling': MPCNNNoPerDimNoMultiPooling,
                'mpcnn_no_per_dim_no_multi_pooling_no_horiz': MPCNNNoPerDimNoMultiPoolingNoHoriz,
                'mpcnn_no_per_dim_no_multi_pooling_no_horiz_no_inf': MPCNNNoPerDimNoMultiPoolingNoHorizNoInf,
                'mpcnn_lite_multichannel': MPCNNLiteMultichannel,
                'mpcnn_lite_multichannel_attention': MPCNNLiteMultiChannelAttention
            }

            filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
            if args.arch in ('mpcnn_lite_multichannel', ):
                model = model_map[args.arch](args.word_vectors_dim, args.holistic_filters, args.per_dim_filters,
                                             filter_widths,
                                             args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, ext_feats,
                                             args.attention, args.wide_conv, args.multichannel)
            else:
                model = model_map[args.arch](args.word_vectors_dim, args.holistic_filters, args.per_dim_filters, filter_widths,
                                              args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, ext_feats,
                                              args.attention, args.wide_conv)
        elif args.arch == 'smcnn':
            model = SMCNN(args.word_vectors_dim, args.holistic_filters, args.max_window_size, args.hidden_units,
                          dataset_cls.NUM_CLASSES, args.dropout, ext_feats, args.attention, args.wide_conv)
        elif args.arch == 'smcnn_with_comp':
            model = SMCNNWithComp(args.word_vectors_dim, args.holistic_filters, args.max_window_size, args.hidden_units,
                          dataset_cls.NUM_CLASSES, args.dropout, ext_feats, args.attention, args.wide_conv)
        elif args.arch == 'smcnn_multi_window':
            filter_widths = list(range(1, args.max_window_size + 1))
            model = SMCNNMultiWindow(args.word_vectors_dim, args.holistic_filters, filter_widths, args.hidden_units,
                          dataset_cls.NUM_CLASSES, args.dropout, ext_feats, args.attention, args.wide_conv)
        else:
            raise ValueError('Unrecognized model variant')

        return model
