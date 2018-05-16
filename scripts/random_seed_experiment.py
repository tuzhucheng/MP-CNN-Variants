import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

import task_queue.tasks as tasks


def run(group, dataset, count):
    device = 0
    if dataset in ('trecqa', 'wikiqa'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
    elif dataset in ('sick', ):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')

    if dataset == 'trecqa':
        epochs = 5
        reg = 0.0005
        dropout = 0.5
        eps = 0.1
        lr = 0.001
    elif dataset == 'wikiqa':
        epochs = 10
        reg = 0.02
        dropout = 0.5
        eps = 1e-8
        lr = 0.0004
    elif dataset == 'sick':
        epochs = 19
        reg = 0.0001
        dropout = 0
        eps = 1e-7
        lr = 0.001

    for _ in range(count):
        gpu = device % 2
        device += 1
        randid = randint(1, 1000000)

        model = f"id_{randid}.castor"
        command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid}"

        print("Running: " + command)
        tasks.run_model.apply_async(args=[command.split(' '),
            group, train_extract, dev_extract, test_extract],
            queue=f'gpu{gpu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check performance of mode across different random seeds')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('dataset', type=str, choices=['trecqa', 'wikiqa', 'sick'], help='dataset')
    parser.add_argument('--count', type=int, default=30, help='number of times to run')
    args = parser.parse_args()
    run(args.group, args.dataset, args.count)
