import math
import os
import pathlib
import shutil

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
    return [1 - sim, sim]


class MSRP(CastorPairDataset):
    NAME = 'msrp'
    NUM_CLASSES = 2
    ID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, postprocessing=Pipeline(get_class_probs))
    RAW_TEXT_FIELD = RawField()

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a MSRP dataset instance
        """
        super(MSRP, self).__init__(path)

    @classmethod
    def _read_file(cls, fn):
        lines = []
        with open(fn, 'r') as f:
            for line in f:
                lines.append(line)
        return lines

    @classmethod
    def splits(cls, path, train='train', test='test', **kwargs):
        # Create temporary files to split train into train and dev
        train_tmp, dev_tmp = f'{train}-tmp', 'dev-tmp'
        pathlib.Path(os.path.join(path, train_tmp)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(path, dev_tmp)).mkdir(parents=True, exist_ok=True)

        train_id = cls._read_file(os.path.join(path, train, 'id.txt'))
        train_a_toks = cls._read_file(os.path.join(path, train, 'a.toks'))
        train_b_toks = cls._read_file(os.path.join(path, train, 'b.toks'))
        train_sim = cls._read_file(os.path.join(path, train, 'sim.txt'))

        dev_lines = np.random.choice(np.arange(len(train_id)), size=400, replace=False)

        train_tmp_id_path = os.path.join(path, train_tmp, 'id.txt')
        train_tmp_sim_path = os.path.join(path, train_tmp, 'sim.txt')
        train_tmp_a_toks = os.path.join(path, train_tmp, 'a.toks')
        train_tmp_b_toks = os.path.join(path, train_tmp, 'b.toks')
        dev_tmp_id_path = os.path.join(path, dev_tmp, 'id.txt')
        dev_tmp_sim_path = os.path.join(path, dev_tmp, 'sim.txt')
        dev_tmp_a_toks = os.path.join(path, dev_tmp, 'a.toks')
        dev_tmp_b_toks = os.path.join(path, dev_tmp, 'b.toks')

        counter = 0

        with open(train_tmp_id_path, 'w') as tid, open(train_tmp_sim_path, 'w') as tsim, open(train_tmp_a_toks, 'w') as ta, open(train_tmp_b_toks, 'w') as tb,\
                open(dev_tmp_id_path, 'w') as did, open(dev_tmp_sim_path, 'w') as dsim, open(dev_tmp_a_toks, 'w') as da, open(dev_tmp_b_toks, 'w') as db:
            for i, (pid, sa, sb, sim) in enumerate(zip(train_id, train_a_toks, train_b_toks, train_sim)):
                counter += 1
                if i in dev_lines:
                    did.write(pid)
                    dsim.write(sim)
                    da.write(sa)
                    db.write(sb)
                else:
                    tid.write(pid)
                    tsim.write(sim)
                    ta.write(sa)
                    tb.write(sb)

        split_results = super(MSRP, cls).splits(path, train=train_tmp, validation=dev_tmp, test=test, **kwargs)

        shutil.rmtree(os.path.join(path, train_tmp))
        shutil.rmtree(os.path.join(path, dev_tmp))

        return split_results

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, validation, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, validation, test, vectors=vectors)

        return BucketIterator.splits((train, validation, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
