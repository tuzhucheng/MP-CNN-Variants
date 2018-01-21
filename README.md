# MP-CNN Variations

This is a PyTorch implementation of MP-CNN as a base model with modifications and additions such as attention and sparse features.

Note, the `master` version of the code current does not implement attention after the torchtext refactoring.
If you are looking for an implementation with attention, checkout [Release v1.0](https://github.com/tuzhucheng/MP-CNN-Variants/releases/tag/v1.0).

Here is the MP-CNN paper:

* Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf). *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

The SICK and MSRVID datasets are available in https://github.com/castorini/data, as well as the GloVe word embeddings.

Directory layout should be like this:
```
├── MP-CNN-Variants
│   ├── README.md
│   ├── ...
├── data
│   ├── README.md
│   ├── ...
│   ├── msrvid/
│   ├── sick/
│   └── GloVe/
```

## SICK Dataset

To run MP-CNN on the SICK dataset mimicking the original paper as closely as possible, use the following command:

```
python main.py mpcnn.sick.model --dataset sick --epochs 19 --epsilon 1e-7 --dropout 0
```

Note the original paper doesn't use dropout, so dropout = 0 mimics this behaviour.

You should be able to obtain Pearson's p to be 0.8767 and Spearman's r to be 0.8147, comparable to the results obtained in the paper (0.8686 and 0.8047).

## MSRVID Dataset

To run MP-CNN on the MSRVID dataset, use the following command:
```
python main.py mpcnn.msrvid.model --dataset msrvid --batch-size 16 --epsilon 1e-7 --epochs 32 --dropout 0 --regularization 0.0025
```

You should be able to obtain Pearson's p to be 0.8893, for reference the performance in the paper is 0.9090.

## TrecQA Dataset

To run MP-CNN on TrecQA, you first need to the `get_trec_eval.sh` script in `utils`.

Then, you can run:
```
python main.py mpcnn.trecqa.model --dataset trecqa --epochs 5 --regularization 0.0005 --dropout 0.5 --eps 0.1
```

You should be able to get a map (mean average precision) of 0.7838 and mrr (mean reciprocal rank) of 0.8228.

These are not the optimal hyperparameters but they are decent. This README will be updated with more optimal hyperparameters and results in the future.

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python main.py mpcnn.wikiqa.model --epochs 10 --dataset wikiqa --batch-size 64 --lr 0.0004 --regularization 0.02
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.693  | 0.709  |
| PyTorch using above config       | 0.6512  | 0.6624 |

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

These are not the optimal hyperparameters but they are decent. This README will be updated with more optimal hyperparameters and results in the future.

To see all options available, use
```
python main.py --help
```

## Dependencies

The model is written in PyTorch. We optionally make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard) to visualize the training process. Just add `--tensorboard` to enable.
