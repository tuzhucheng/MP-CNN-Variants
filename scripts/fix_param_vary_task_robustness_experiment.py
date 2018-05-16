import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

import task_queue.tasks as tasks


def run(group, arch_group, datasets, count):
    dropout = 0
    device = 2
    for _ in range(count):
        lr = round(10 ** uniform(-4, -3), 5)
        reg = round(10 ** uniform(-5, -3), 7)
        randid = randint(1, 1000000)
        print('Seed', randid)
        if arch_group == 'conv':
            archs = ['mpcnn', 'mpcnn_holistic_only']
        elif arch_group == 'pool':
            archs = ['mpcnn', 'mpcnn_pool_max_only', 'mpcnn_pool_mean_sym', 'mpcnn_pool_no_mean_sym']
        elif arch_group == 'comp':
            archs = ['mpcnn', 'mpcnn_comp_horiz_only', 'mpcnn_comp_vert_only', 'mpcnn_comp_vert_holistic_only']
        elif arch_group == 'comp-dist':
            archs = ['mpcnn', 'mpcnn_comp_cosine', 'mpcnn_comp_euclidean', 'mpcnn_comp_abs_diff']
        elif arch_group == 'conv-pool':
            archs = ['mpcnn', 'mpcnn_holistic_pool_max_only']
        elif arch_group == 'window':
            archs = ['mpcnn', 'mpcnn_single_window', 'mpcnn_single_window_with_inf']
        elif arch_group == 'lite':
            archs = ['mpcnn', 'mpcnn_no_per_dim_no_multi_pooling']
        elif arch_group == 'wideconv':
            archs = ['mpcnn_no_per_dim_no_multi_pooling']
        elif arch_group == 'sparse':
            archs = ['mpcnn', 'smcnn']
        elif arch_group == 'attention':
            archs = ['mpcnn_no_per_dim_no_multi_pooling']
        elif arch_group == 'conv2d':
            archs = ['mpcnn', 'mpcnn_lite_multichannel']
        elif arch_group == 'ind-filters':
            archs = ['mpcnn', 'mpcnn_independent_filters']
        elif arch_group == 'window-size':
            archs = ['mpcnn_no_per_dim_no_multi_pooling']
        elif arch_group == 'multichannel':
            archs = ['mpcnn_lite_multichannel']

        for arch in archs:
            for dataset in datasets:
                if dataset in ('trecqa', 'wikiqa'):
                    train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
                    dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
                    test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
                elif dataset in ('sick', ):
                    train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
                    dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
                    test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')

                dropout = 0
                eps = 1e-8
                if dataset == 'trecqa':
                    prob = uniform(0, 1)
                    gpu = 1 if prob < 0.5 else 2
                    epochs = 5
                elif dataset == 'wikiqa':
                    prob = uniform(0, 1)
                    gpu = 1 if prob < 0.5 else 2
                    epochs = 10
                elif dataset == 'sick':
                    prob = uniform(0, 1)
                    gpu = 1 if prob < 0.5 else 2
                    epochs = 19

                model = f"arch_{arch}_dataset_{dataset}_seed_{randid}"
                command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model} --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid}"

                if arch_group not in ('window-size', ):
                    print(command)
                    tasks.run_model.apply_async(args=[command.split(' '),
                        group, train_extract, dev_extract, test_extract],
                        queue=f'gpu{gpu}')

                if arch_group == 'wideconv':
                    command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model}_wideconv --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --wide-conv"
                    print(command)
                    tasks.run_model.apply_async(args=[command.split(' '),
                        group, train_extract, dev_extract, test_extract],
                        queue=f'gpu{gpu}')
                elif arch_group == 'sparse':
                    command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model}_sparse --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --sparse-features"
                    print(command)
                    tasks.run_model.apply_async(args=[command.split(' '),
                        group, train_extract, dev_extract, test_extract],
                        queue=f'gpu{gpu}')
                elif arch_group == 'attention':
                    command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model}_attn_basic --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --attention basic"
                    command2 = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model}_attn_idf --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --attention idf"
                    commands = [command, command2]
                    for command in commands:
                        print(command)
                        tasks.run_model.apply_async(args=[command.split(' '),
                            group, train_extract, dev_extract, test_extract],
                            queue=f'gpu{gpu}')
                elif arch_group == 'window-size':
                    for w in range(1, 7):
                        command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model}_wideconv_w{w} --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --wide-conv --max-window-size {w}"
                        print(command)
                        tasks.run_model.apply_async(args=[command.split(' '),
                            group, train_extract, dev_extract, test_extract],
                            queue=f'gpu{gpu}')
                elif arch_group == 'multichannel':
                    command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model}_multichannel --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --multichannel"
                    print(command)
                    tasks.run_model.apply_async(args=[command.split(' '),
                        group, train_extract, dev_extract, test_extract],
                        queue=f'gpu{gpu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix hyperparameters, vary dataset, model robustness experiment')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('--arch-group', type=str, choices=['conv', 'pool', 'comp', 'comp-dist', 'conv-pool', 'window', 'sparse', 'attention', 'lite', 'conv2d', 'wideconv', 'ind-filters', 'window-size', 'multichannel'], help='model architecture group')
    parser.add_argument('--datasets', nargs='+', choices=['trecqa', 'wikiqa', 'sick'], help='datasets')
    parser.add_argument('--count', type=int, default=5, help='number of times to run')
    args = parser.parse_args()
    run(args.group, args.arch_group, args.datasets, args.count)
