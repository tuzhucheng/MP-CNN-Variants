import argparse
from random import choice, randint, uniform
import re
import json
import random
from subprocess import call

import task_queue.tasks as tasks
from utils.hyperband import Hyperband

device = 0

def get_random_configuration():
    learning_rate = round(10 ** uniform(-4, -3), 5)
    filters = choice([50, 100, 150, 300])
    reg = round(10 ** uniform(-5, -3), 7)
    return {
        'lr': learning_rate,
        'filters': filters,
        'reg': reg
    }

def run_and_return_eval(num_iters, expt_group, dataset, log_interval, params):
    global device
    gpu = device % 2
    device += 1

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

    randid = random.randint(1, 1000000)
    lr = params['lr']
    filters = params['filters']
    reg = params['reg']

    model_name = f"lr_{lr}_f_{filters}_reg_{reg}_id_{randid}.castor"
    command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model_name} --arch smcnn --dataset {dataset} --log-interval {log_interval} --epochs {num_iters} --device {gpu} --holistic-filters {filters} --batch-size 64 --lr {lr} --regularization {reg}"

    print("Running: " + command)
    res_future = tasks.run_model.apply_async(args=[command.split(' '),
        expt_group, train_extract, dev_extract, test_extract],
        queue=f'gpu{gpu}')

    return res_future

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper parameters sweeper')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('dataset', type=str, choices=['trecqa', 'wikiqa', 'sick', 'msrvid'], help='dataset')
    parser.add_argument('metric', type=str, choices=['map', 'pearson'], help='metric to optimize')
    parser.add_argument('--max_iters', type=int, default=81, help='Hyperband Max Iters')
    parser.add_argument('--eta', type=int, default=3, help='Hyperband Iters')
    parser.add_argument('--log-interval', type=int, default=1000, help='log interval')
    args = parser.parse_args()

    hb = Hyperband(get_random_configuration, run_and_return_eval, args.max_iters, args.eta)
    hb.run(args.group, args.dataset, args.metric, args.log_interval)
