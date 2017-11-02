import os
import re
import subprocess
import time

import torch
import torch.nn.functional as F

from evaluators.evaluator import Evaluator


class TRECQAEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(TRECQAEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        qids = []
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            qids.extend(batch.id.data.cpu().numpy())
            output = self.model(batch.a, batch.b, batch.ext_feats)
            test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).data[0]

            true_labels.append(batch.label.data.cpu())
            predictions.append(output.data.exp()[:, 1])

            del output

        qids = list(map(lambda n: int(round(n * 10, 0)) / 10, qids))
        predictions = torch.cat(predictions).cpu().numpy()
        true_labels = torch.cat(true_labels).cpu().numpy()

        qrel_fname = 'trecqa_{}_{}.qrel'.format(time.time(), self.data_loader.device)
        results_fname = 'trecqa_{}_{}.results'.format(time.time(), self.data_loader.device)
        qrel_template = '{qid} 0 {docno} {rel}\n'
        results_template = '{qid} 0 {docno} 0 {sim} mpcnn\n'
        with open(qrel_fname, 'w') as f1, open(results_fname, 'w') as f2:
            docnos = range(len(qids))
            for qid, docno, predicted, actual in zip(qids, docnos, predictions, true_labels):
                f1.write(qrel_template.format(qid=qid, docno=docno, rel=actual))
                f2.write(results_template.format(qid=qid, docno=docno, sim=predicted))

        trec_out = subprocess.check_output(['./utils/trec_eval-9.0.5/trec_eval', '-m', 'map', '-m', 'recip_rank', qrel_fname, results_fname])
        trec_out_lines = str(trec_out, 'utf-8').split('\n')
        mean_average_precision = float(trec_out_lines[0].split('\t')[-1])
        mean_reciprocal_rank = float(trec_out_lines[1].split('\t')[-1])

        test_cross_entropy_loss /= len(batch.dataset.examples)

        os.remove(qrel_fname)
        os.remove(results_fname)

        return [test_cross_entropy_loss, mean_average_precision, mean_reciprocal_rank], ['cross entropy loss', 'map', 'mrr']
