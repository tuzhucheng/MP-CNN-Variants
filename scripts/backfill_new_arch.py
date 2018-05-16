import argparse
from random import choice, randint, uniform
import re
import json
from subprocess import call

from db import conn
import task_queue.tasks as tasks


def run(group, arch, datasets, sparse, attention):
    dropout = 0

    cursor = conn.cursor()
    # takes one entry per seed (configuration)
    experiment_rows = [json.loads(r[0]) for r in cursor.execute("""
        select args from experiments e
        join experiment_groups g on e.group_id=g.gid
        where g.name=? group by json_extract(args, '$.seed')
        having min(e.rowid) order by e.rowid""", (group,))]

    for d in experiment_rows:

        if 'lr' not in d or 'regularization' not in d or 'dropout' not in d or 'eps' not in d or 'seed' not in d:
            import ipdb; ipdb.set_trace()
        d.pop('program')
        d.pop('main')
        d['arg2'] = d['arg2'].replace('arch_' + d['arch'], 'arch_{}')
        d.pop('arch')
        d['arg2'] = d['arg2'].replace('dataset_' + d['dataset'], 'dataset_{}')
        d.pop('dataset')
        d.pop('epochs', None)
        d.pop('device', None)

    for config in experiment_rows:
        for dataset in datasets:
            if dataset in ('trecqa', 'wikiqa'):
                train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
                dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
                test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<map>\d+\.\d+)\s+(?P<mrr>\d+\.\d+)\s+(?P<cross_entropy_loss>\d+\.\d+)')
            elif dataset in ('sick', ):
                train_extract = ('stderr', r'INFO\s+-\s+train\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
                dev_extract = ('stderr', r'INFO\s+-\s+dev\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')
                test_extract = ('stderr', r'INFO\s+-\s+test\s+(?P<pearson>\d+\.\d+)\s+(?P<spearman>\d+\.\d+)\s+(?P<kl_div>\d+\.\d+)')

            if dataset == 'trecqa':
                gpu = 1
                epochs = 5
            elif dataset == 'wikiqa':
                gpu = 2
                epochs = 10
            elif dataset == 'sick':
                gpu = 2
                epochs = 19

            model = config['arg2'].format(arch, dataset)
            lr = config['lr']
            reg = config['regularization']
            dropout = config['dropout']
            eps = config['eps']
            seed = config['seed']
            command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py {model} --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {seed}"
            if sparse:
                command += ' --sparse-features'

            if attention is not None and attention != 'none':
                command += f' --attention {attention}'

            print(command)
            tasks.run_model.apply_async(args=[command.split(' '),
                group, train_extract, dev_extract, test_extract],
                queue=f'gpu{gpu}')

            # elif arch_group == 'attention':
                # command = f"python /u/z3tu/castorini/MP-CNN-Variants/main.py /u/z3tu/castorini/MP-CNN-Variants/saved_models/{model} --arch {arch} --dataset {dataset} --epochs {epochs} --device {gpu} --lr {lr} --regularization {reg} --dropout {dropout} --eps {eps} --seed {randid} --attention"
                # print(command)
                # tasks.run_model.apply_async(args=[command.split(' '),
                    # group, train_extract, dev_extract, test_extract],
                    # queue=f'gpu{gpu}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill fix seed and hyperparameter experiments with new architecture')
    parser.add_argument('group', type=str, help='experiment group')
    parser.add_argument('arch', type=str, help='architecture')
    parser.add_argument('--datasets', nargs='+', choices=['trecqa', 'wikiqa', 'sick'], help='datasets')
    parser.add_argument('--sparse', action='store_true', default=False, help='sparse')
    parser.add_argument('--attention', help='attention')
    args = parser.parse_args()
    run(args.group, args.arch, args.datasets, args.sparse, args.attention)
