import math

import numpy as np
import torch
from torchtext.data.field import Field, RawField
from torchtext.data.iterator import BucketIterator
from torchtext.data.pipeline import Pipeline
from torchtext.vocab import Vectors

from datasets.castor_dataset import CastorPairDataset


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(STS.NUM_CLASSES)
    ceil, floor = math.ceil(sim), math.floor(sim)
    if ceil == floor:
        class_probs[floor] = 1
    else:
        class_probs[floor] = ceil - sim
        class_probs[ceil] = sim - floor

    return class_probs


class STS(CastorPairDataset):
    NAME = 'sts'
    NUM_CLASSES = 6
    ID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(dtype=torch.float32, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, dtype=torch.float32, use_vocab=False, batch_first=True, postprocessing=Pipeline(get_class_probs))
    RAW_TEXT_FIELD = RawField()

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a STS dataset instance
        """
        super(STS, self).__init__(path)

    @classmethod
    def splits(cls, path, train='train', validation='dev', test='test', **kwargs):
        return super(STS, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, device, batch_size=64, shuffle=True, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to word vectors file
        :param device: PyTorch device
        :param batch_size: batch size
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
