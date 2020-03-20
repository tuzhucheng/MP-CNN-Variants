import os

import torch
import torch.nn as nn

from datasets.sick import SICK
from datasets.msrp import MSRP
from datasets.msrvid import MSRVID
from datasets.trecqa import TRECQA
from datasets.wikiqa import WikiQA
from datasets.sts import STS


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].normal_(0, 0.01)
        return cls.cache[size_tup]


class MPCNNDatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device):
        trec_eval_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils/trec_eval-9.0.7/trec_eval')
        if dataset_name == 'sick':
            dataset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'datasets', 'sick/')
            train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, device, batch_size, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(SICK.TEXT_FIELD.vocab.vectors)
            return SICK, embedding, train_loader, test_loader, dev_loader
        if dataset_name == 'sts':
            dataset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'datasets', 'sts/')
            train_loader, dev_loader, test_loader = STS.iters(dataset_root, word_vectors_file, word_vectors_dir, device, batch_size, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(STS.TEXT_FIELD.vocab.vectors)
            return STS, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'msrp':
            dataset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'datasets', 'msrp/')
            train_loader, dev_loader, test_loader = MSRP.iters(dataset_root, word_vectors_file, word_vectors_dir, device, batch_size, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(MSRP.TEXT_FIELD.vocab.vectors)
            return MSRP, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'msrvid':
            dataset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'datasets', 'msrvid/')
            dev_loader = None
            train_loader, test_loader = MSRVID.iters(dataset_root, word_vectors_file, word_vectors_dir, device, batch_size, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(MSRVID.TEXT_FIELD.vocab.vectors)
            return MSRVID, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'trecqa':
            if not os.path.exists(trec_eval_path):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside utils/ (as working directory) before continuing.')
            dataset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'datasets', 'TrecQA/')
            train_loader, dev_loader, test_loader = TRECQA.iters(dataset_root, word_vectors_file, word_vectors_dir, device, batch_size, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(TRECQA.TEXT_FIELD.vocab.vectors)
            return TRECQA, embedding, train_loader, test_loader, dev_loader
        elif dataset_name == 'wikiqa':
            if not os.path.exists(trec_eval_path):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'datasets', 'WikiQA/')
            train_loader, dev_loader, test_loader = WikiQA.iters(dataset_root, word_vectors_file, word_vectors_dir, device, batch_size, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(WikiQA.TEXT_FIELD.vocab.vectors)
            return WikiQA, embedding, train_loader, test_loader, dev_loader
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))
