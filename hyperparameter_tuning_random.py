import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

import task_queue.tasks as tasks


def run(group, dataset, count, epochs, log_interval):
    device = 0
    if dataset in ('trecqa', 'wikiqa'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<cross_entropy_loss>\d+\.\d+)\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<cross_entropy_loss>\d+\.\d+)\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<cross_entropy_loss>\d+\.\d+)\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)')
    elif dataset in ('sick'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
    else:
        raise ValueError('')

    for _ in range(count):
        learning_rate = round(10 ** uniform(-4, -3), 5)
        filters = choice([50, 100, 150, 300])
        reg = round(10 ** uniform(-5, -3), 7)
        gpu = device % 2
        device += 1
        randid = randint(1, 1000000)

        model_name = f"lr_{learning_rate}_f_{filters}_reg_{reg}_id_{randid}.castor"
        command = "python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model} --arch smcnn --dataset {dataset} --log-interval {log_interval} --epochs {epo} --device {dev} --holistic-filters {filters} --batch-size 64 --lr {learning_rate} --regularization {reg}".format(epo=epochs, model=model_name, dataset=dataset, dev=gpu, log_interval=log_interval, learning_rate=learning_rate, filters=filters, reg=reg)

        print("Running: " + command)
        tasks.run_model.apply_async(args=[command.split(' '),
            group, train_extract, dev_extract, test_extract],
            queue=f'gpu{gpu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper parameters sweeper')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('dataset', type=str, choices=['trecqa', 'wikiqa', 'sick', 'msrvid'], help='dataset')
    parser.add_argument('--count', type=int, default=30, help='number of times to run')
    parser.add_argument('--epochs', type=int, default=7, help='number of epochs to run')
    parser.add_argument('--log-interval', type=int, default=1000, help='log interval')
    args = parser.parse_args()
    run(args.group, args.dataset, args.count, args.epochs, args.log_interval)
