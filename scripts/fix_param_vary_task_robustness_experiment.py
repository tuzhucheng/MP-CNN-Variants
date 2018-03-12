import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

import task_queue.tasks as tasks


def run(group, arch_group, datasets, count):
    dropout = 0
    device = 1
    for _ in range(count):
        lr = round(10 ** uniform(-4, -3), 5)
        reg = round(10 ** uniform(-5, -3), 7)
        randid = randint(1, 1000000)
        print('Seed', randid)
        if arch_group == 'pool':
            archs = ['mpcnn', 'mpcnn_pool_max_only', 'mpcnn_pool_mean_sym', 'mpcnn_pool_no_mean_sym']
        else:
            archs = ['mpcnn', 'mpcnn_comp_horiz_only', 'mpcnn_comp_vert_only', 'mpcnn_comp_unit1_only', 'mpcnn_comp_unit2_only']

        for arch in archs:
            for dataset in datasets:
                if dataset in ('trecqa', 'wikiqa'):
                    train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<cross_entropy_loss>\d+\.\d+)\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)')
                    dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<cross_entropy_loss>\d+\.\d+)\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)')
                    test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<cross_entropy_loss>\d+\.\d+)\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)')
                elif dataset in ('sick', ):
                    train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
                    dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
                    test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')

                dropout = 0
                eps = 1e-8
                if dataset == 'trecqa':
                    gpu = device
                    device = 1 if device == 2 else 2
                    epochs = 5
                elif dataset == 'wikiqa':
                    gpu = 0
                    epochs = 10
                elif dataset == 'sick':
                    gpu = 0
                    epochs = 19

                model = f"arch_{arch}_dataset_{dataset}_seed_{randid}.castor"
                command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model} --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid}"

                print(command)
                tasks.run_model.apply_async(args=[command.split(' '),
                    group, train_extract, dev_extract, test_extract],
                    queue=f'gpu{gpu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix hyperparameters, vary dataset, model robustness experiment')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('--arch-group', type=str, choices=['pool', 'comp'], help='model architecture group')
    parser.add_argument('--datasets', nargs='+', choices=['trecqa', 'wikiqa', 'sick'], help='datasets')
    parser.add_argument('--count', type=int, default=5, help='number of times to run')
    args = parser.parse_args()
    run(args.group, args.arch_group, args.datasets, args.count)
