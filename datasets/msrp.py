import os
import pathlib
import re
import shutil
import uuid

import numpy as np
import torch
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.field import Field, RawField
from torchtext.data.iterator import BucketIterator
from torchtext.data.pipeline import Pipeline
from torchtext.vocab import Vectors

from datasets.idf_utils import get_pairwise_word_to_doc_freq


class MSRP(Dataset):
    NAME = 'msrp'
    NUM_CLASSES = 2
    EXT_FEATS = 6
    ID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it
    EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    RAW_TEXT_FIELD = RawField()

    NUMBER_PATTERN = re.compile(r'((\d+,)*\d+\.?\d*)')

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a MSRP dataset instance
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD), ('ext_feats', self.EXT_FEATS_FIELD),
                ('label', self.LABEL_FIELD), ('sentence_1_raw', self.RAW_TEXT_FIELD), ('sentence_2_raw', self.RAW_TEXT_FIELD)]

        examples = []
        with open(os.path.join(path, 'a.toks'), 'r') as f1, open(os.path.join(path, 'b.toks'), 'r') as f2:
            sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
            sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        self.word_to_doc_cnt = word_to_doc_cnt

        with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
            for pair_id, l1, l2, label in zip(id_file, sent_list_1, sent_list_2, label_file):
                pair_id = pair_id.rstrip('.\n')
                label = label.rstrip('.\n')
                ext_feats = []

                # Number features
                sent1_nums, sent2_nums = [], []
                match = self.NUMBER_PATTERN.search(' '.join(l1))
                if match:
                    for g in match.groups():
                        if g is not None:
                            sent1_nums.append(g)

                match = self.NUMBER_PATTERN.search(' '.join(l2))
                if match:
                    for g in match.groups():
                        if g is not None:
                            sent2_nums.append(g)

                sent1_nums = set(sent1_nums)
                sent2_nums = set(sent2_nums)
                exact = int(sent1_nums == sent2_nums)
                superset = int(sent1_nums.issuperset(sent2_nums) or sent2_nums.issuperset(sent1_nums))
                ext_feats.append(1 if (exact or (len(sent1_nums) == 0 and len(sent2_nums) == 0)) else 0)
                ext_feats.append(exact)
                ext_feats.append(superset)

                # Length difference
                ext_feats.append(len(l2) - len(l1))

                # Overlap
                overlap = len(set(l1) & set(l2))
                ext_feats.append(overlap / len(l1))
                ext_feats.append(overlap / len(l2))

                example = Example.fromlist([pair_id, l1, l2, ext_feats, label, ' '.join(l1), ' '.join(l2)], fields)
                examples.append(example)

        super(MSRP, self).__init__(examples, fields)

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
        uid = uuid.uuid4()
        train_tmp, dev_tmp = f'{train}-tmp-{uid}', f'dev-tmp-{uid}'
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
