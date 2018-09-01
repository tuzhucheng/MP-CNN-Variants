# MP-CNN Variations

This is a PyTorch implementation of MP-CNN as a base model with modifications and additions such as attention and sparse features.

Here is the MP-CNN paper:

* Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf). *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

The datasets are available in https://git.uwaterloo.ca/jimmylin/Castor-data, as well as the GloVe word embeddings.

Directory layout should be like this:
```
├── MP-CNN-Variants
│   ├── README.md
│   ├── ...
├── Castor-data
│   ├── README.md
│   ├── ...
│   ├── msrvid/
│   ├── sick/
│   └── GloVe/
```

Note the original paper doesn't use dropout, so dropout=0 mimics this behaviour to allow for fair comparison in the results reported below.

## SICK Dataset

To run MP-CNN on the SICK dataset mimicking the original paper as closely as possible, use the following command:

```
python main.py mpcnn.sick.model --dataset sick --epochs 19 --dropout 0 --lr 0.0005
```

| Implementation and config        | Pearson's r    | Spearman's p    | MSE    |
| -------------------------------- |:--------------:|:---------------:|:------:|
| Paper                            | 0.8686         | 0.8047          | 0.2606 |
| PyTorch using above config       | 0.8738         | 0.8116          | 0.2414 |


## TrecQA Dataset

To run MP-CNN on TrecQA, you first need to run the `get_trec_eval.sh` script in `utils`.

Then, you can run:
```
python main.py mpcnn.trecqa.model --arch mpcnn --dataset trecqa --epochs 5 --holistic-filters 200 --lr 0.00018 --regularization 0.0006405 --dropout 0
```

| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.762  | 0.854  |
| PyTorch using above config       | 0.771  | 0.823  |

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## WikiQA Dataset

You also need `trec_eval` for this dataset, similar to TrecQA.

Then, you can run:
```
python main.py mpcnn.wikiqa.model --arch mpcnn --dataset wikiqa --epochs 5 --holistic-filters 100 --lr 0.00042 --regularization 0.0001683 --dropout 0
```
| Implementation and config        | map    | mrr    |
| -------------------------------- |:------:|:------:|
| Paper                            | 0.693  | 0.709  |
| PyTorch using above config       | 0.709  | 0.721  |

The paper results are reported in [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2983872).

## Other Datasets

### MSRVID

To run MP-CNN on the MSRVID dataset, use the following command:
```
python main.py mpcnn.msrvid.model --dataset msrvid --batch-size 16 --epsilon 1e-7 --epochs 32 --dropout 0 --regularization 0.0025
```

You should be able to obtain Pearson's p to be 0.8989 (untuned), for reference the performance in the paper is 0.9090.

### MSRP Dataset

To run MP-CNN on the MSRP dataset, use the following command:

```
python main.py mpcnn.msrp.model --dataset msrp --epochs 15
```

To see all options available, use
```
python main.py --help
```

## Dependencies

The model is written in PyTorch. We optionally make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard) to visualize the training process. Just add `--tensorboard` to enable.

## Experimental

There are some scripts in this repo for hyperparameter optimization using [watermill](https://github.com/tuzhucheng/watermill) with some hacks since the library is in alpha. Hence, the imports in `hyperparameter_tuning_{random,hyperband}.py` and `utils/hyperband.py` will not work for you at the moment.

## References

For results, please see my Master's thesis [here](https://uwspace.uwaterloo.ca/handle/10012/13297):

```
@mastersthesis{tu2018experimental,
  title={An Experimental Analysis of Multi-Perspective Convolutional Neural Networks},
  author={Tu, Zhucheng},
  year={2018},
  school={University of Waterloo}
}
```
