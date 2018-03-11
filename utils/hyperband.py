"""
Implementation of "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
https://arxiv.org/abs/1603.06560
Adapted from code on blog post: https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
"""
import logging
from math import ceil, log
import pprint

from celery.result import ResultSet
import numpy as np


class Hyperband(object):

    def __init__(self, get_random_hyperparameter_configuration, run_then_return_eval, fixed_args, max_iter=81, eta=3):
        self.get_random_hyperparameter_configuration = get_random_hyperparameter_configuration
        self.run_then_return_eval = run_then_return_eval
        self.fixed_args = fixed_args
        self.max_iter = max_iter  # max epochs per configuration
        self.eta = eta  # down-sampling rate
        # total number of unique executions of Successive Halving (minus 1), just log_{eta} (max_iter)
        self.s_max = int(log(self.max_iter) / log(self.eta))
        # total number of iterations per execution of Successive Halving
        self.B = (self.s_max + 1) * self.max_iter

        self.best_metric = np.inf

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.logger = logger

    def run(self, expt_group, arch, dataset, metric, log_interval):
        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s+1) * (self.eta**s)))
            # initial number of iterations to run configurations for
            r = self.max_iter * self.eta**(-s)

            # Successive Halving with (n, r)
            T = [self.get_random_hyperparameter_configuration() for _ in range(n)]

            for i in range(s+1):
                # Run each of the n_i configs for r_i iters and keep best n_i / eta
                n_i = n * self.eta**(-i)
                r_i = int(r * self.eta**i)

                self.logger.info('%s configurations, %s iterations each', len(T), r_i)

                result_set = []

                for t in T:
                    for k, v in self.fixed_args.items():
                        t[k] = v
                    async_res = self.run_then_return_eval(r_i, expt_group, arch, dataset, log_interval, t)
                    result_set.append(async_res)

                result_set = ResultSet(result_set)
                result_set.join_native()

                self.logger.info('Completed new round')
                results = []
                for async_res in result_set:
                    res = async_res.get()
                    result = {
                        'lr': res['args']['lr'],
                        'filters': res['args']['holistic_filters'],
                        'reg': res['args']['regularization']
                    }
                    result[metric] = res['dev'][metric]
                    results.append(result)

                    if result[metric] > self.best_metric:
                        self.best_metric = result[metric]
                        self.logger.info('This is a new best...')

                sorted_results = sorted(results, key=lambda d: d[metric], reverse=True)
                for d in sorted_results:
                    d.pop(metric)

                T = sorted_results[:int(n_i) // self.eta]

