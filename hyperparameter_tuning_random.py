import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

import task_queue.tasks as tasks


def run(group, dataset, count, epochs, log_interval, gpu):
    if dataset in ('trecqa', 'wikiqa'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
    elif dataset in ('sick'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
    elif dataset in ('msrp'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<accuracy>\d+\.\d+)\s+(?P<f1>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<accuracy>\d+\.\d+)\s+(?P<f1>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<accuracy>\d+\.\d+)\s+(?P<f1>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
    else:
        raise ValueError('')

    gpu = 2
    for _ in range(count):
        learning_rate = round(10 ** uniform(-4, -3), 5)
        optimizer = choice(['adam'])
        momentum = choice([0.3, 0.4, 0.5, 0.6, 0.7])
        filters = choice([50, 100, 200, 300])
        reg = round(1 * 10 ** uniform(-5, -3), 7)
        hidden_units = choice([100, 125, 150, 175, 200, 225, 250, 275, 300])
        dropout = choice([0, 0.1, 0.3, 0.5])
        # dropout = 0
        patience = choice([2, 3, 4, 5])
        lr_reduce = choice([0.3, 0.5, 0.7])
        # device += 1
        randid = randint(1, 1000000)

        model_name = f"ds_{dataset}_lr_{learning_rate}_f_{filters}_reg_{reg}_id_{randid}.castor"
        command = "python main.py saved_models/{model} --arch mpcnn_no_per_dim_no_multi_pooling --optimizer {optimizer} --momentum {momentum} --dataset {dataset} --hidden-units {hidden_units} --log-interval {log_interval} --epochs {epo} --device {dev} --holistic-filters {filters} --batch-size 64 --lr {learning_rate} --regularization {reg} --attention basic --dropout {dropout} --patience {patience} --lr-reduce-factor {lr_reduce} --wide-conv --sparse-features".format(optimizer=optimizer, epo=epochs, model=model_name, dataset=dataset, hidden_units=hidden_units, dev=gpu, log_interval=log_interval, learning_rate=learning_rate, filters=filters, reg=reg, dropout=dropout, patience=patience, lr_reduce=lr_reduce, momentum=momentum)

        print("Running: " + command)
        tasks.run_model.apply_async(args=[command.split(' '),
            group, train_extract, dev_extract, test_extract],
            queue=f'gpu{gpu}')

        prob = uniform(0, 1)
        gpu = 1 if prob > 0.5 else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper parameters sweeper')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('dataset', type=str, choices=['trecqa', 'wikiqa', 'sick', 'msrp'], help='dataset')
    parser.add_argument('--count', type=int, default=30, help='number of times to run')
    parser.add_argument('--epochs', type=int, default=7, help='number of epochs to run')
    parser.add_argument('--log-interval', type=int, default=1000, help='log interval')
    parser.add_argument('--gpu', type=int, default=0, help='gpu to use')
    args = parser.parse_args()
    run(args.group, args.dataset, args.count, args.epochs, args.log_interval, args.gpu)
