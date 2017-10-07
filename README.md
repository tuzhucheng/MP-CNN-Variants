# MP-CNN Variations

This is a PyTorch implementation of MP-CNN as a base model with modifications and additions such as attention and sparse features.

Here is the MP-CNN paper:

* Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf). *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

The SICK and MSRVID datasets are available in https://github.com/castorini/data, as well as the GloVe word embeddings.

Directory layout should be like this:
```
├── Castor
│   ├── README.md
│   ├── ...
│   └── mp_cnn/
├── data
│   ├── README.md
│   ├── ...
│   ├── msrvid/
│   ├── sick/
│   └── GloVe/
```

To run MP-CNN on the SICK dataset mimicking the original paper as closely as possible, use the following command:

```
python main.py mpcnn.sick.model.castor --dataset sick --epochs 19 --epsilon 1e-7 --dropout 0
```

Note the original paper doesn't use dropout, so dropout = 0 mimics this behaviour.

You should be able to obtain Pearson's p to be 0.8744 and Spearman's r to be 0.8183, slightly better than the results obtained in the paper (0.8686 and 0.8047).

To run MP-CNN on the MSRVID dataset, use the following command:
```
python main.py mpcnn.msrvid.model.castor --dataset msrvid --batch-size 16 --epsilon 1e-7 --epochs 32 --dropout 0 --regularization 0.0025
```

You should be able to obtain Pearson's p to be 0.9072, very close to the performance in the paper (0.9090).

These are not the optimal hyperparameters but they are decent. This README will be updated with more optimal hyperparameters and results in the future.

To see all options available, use
```
python main.py --help
```

## Dependencies

The model is written in PyTorch. We optionally make use of https://github.com/lanpa/tensorboard-pytorch to connect to [TensorBoard](https://github.com/tensorflow/tensorboard) to visualize the training process. Just add `--tensorboard` to enable.
