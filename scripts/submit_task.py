import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

import task_queue.tasks as tasks


def run(group, dataset, gpu, command):
    if dataset in ('trecqa', 'wikiqa'):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
    elif dataset in ('sick', ):
        train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
        test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')

    print(command)
    tasks.run_model.apply_async(args=[command.split(' '),
        group, train_extract, dev_extract, test_extract],
        queue=f'gpu{gpu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit task directly')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('dataset', choices=['trecqa', 'wikiqa', 'sick'], help='dataset')
    parser.add_argument('gpu', type=int, help='GPU')
    parser.add_argument('taskstr', type=str, help='Command to run')
    args = parser.parse_args()
    run(args.group, args.dataset, args.gpu, args.taskstr)
