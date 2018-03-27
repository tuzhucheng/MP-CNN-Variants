MP-CNN Lite Results
===================

## TrecQA (clean)

```
python main.py trecqa.mpcnn.lite.model --arch mpcnn_no_per_dim_no_multi_pooling --dataset trecqa --epochs 5 --lr 0.0002 --regularization 0.000126 --wide-conv --attention basic --dropout 0.3 --sparse-features
```

| Split       | MAP     | MRR    |
| ----------- |:-------:| ------:|
| Dev         | 0.8028  | 0.8750 |
| Test        | 0.7903  | 0.8449 |

## WikiQA

```
python main.py wikiqa.mpcnn.lite.model --arch mpcnn_no_per_dim_no_multi_pooling --dataset wikiqa --epochs 5 --lr 0.00038 --regularization 0.0003396 --wide-conv --attention basic --dropout 0.5 --sparse-features
```

| Split       | MAP     | MRR    |
| ----------- |:-------:| ------:|
| Dev         | 0.7552  | 0.7639 |
| Test        | 0.7106  | 0.7270 |
