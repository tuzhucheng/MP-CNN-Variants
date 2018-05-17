MP-CNN Lite Results
===================

Note, the lite model in this doc just means MP-CNN without per-dimensional convolution and multiple types of pooling. It uses >8x fewer parameters as the full model!

## SICK

```
python main.py sick.mpcnn.lite.model --arch mpcnn_no_per_dim_no_multi_pooling --dataset sick --epochs 19 --lr 0.00086 --regularization 0.0002672 --wide-conv --attention basic --dropout 0.5
```

| Split              | Pearson's r     | Spearman's p    |
| ------------------ |:---------------:| ---------------:|
| Full model (paper) | 0.8686          | 0.8047          |
| Lite model         | 0.8805          | 0.8227          |

## TrecQA (clean)

```
python main.py trecqa.mpcnn.lite.model --arch mpcnn_no_per_dim_no_multi_pooling --dataset trecqa --epochs 5 --lr 0.00037 --regularization 0.0017304 --dropout 0
```

| Split              | MAP        | MRR      |
| ------------------ |:----------:| --------:|
| Full model (paper) | 0.762      | 0.854    |
| Lite model         | 0.795      | 0.889    |

## WikiQA

```
python main.py wikiqa.mpcnn.lite.model --arch mpcnn_no_per_dim_no_multi_pooling --dataset wikiqa --epochs 5 --lr 0.00025 --regularization 0.0001088 --wide-conv --attention basic --sparse-features
```

| Split              | MAP        | MRR      |
| ------------------ |:----------:| --------:|
| Full model (paper) | 0.693      | 0.709    |
| Lite model         | 0.696      | 0.708    |
