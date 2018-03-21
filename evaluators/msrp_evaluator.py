from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from evaluators.evaluator import Evaluator


class MSRPEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device):
        super(MSRPEvaluator, self).__init__(dataset_cls, model, embedding, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        test_kl_div_loss = 0
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            # Select embedding
            sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
            sent2 = self.embedding(batch.sentence_2).transpose(1, 2)

            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            test_kl_div_loss += F.kl_div(output, batch.label, size_average=False).data[0]

            true_labels.append(batch.label.data)
            predictions.append(output.exp().data)

            del output

        test_kl_div_loss /= len(batch.dataset.examples)
        predictions = torch.cat(predictions)[:, 1].cpu().numpy()
        predictions = (predictions >= 0.5).astype(int)  # use 0.5 as default threshold
        true_labels = torch.cat(true_labels)[:, 1].cpu().numpy()
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        return [accuracy, f1, test_kl_div_loss], ['accuracy', 'f1', 'KL-divergence loss']

    def get_final_prediction_and_label(self, batch_predictions, batch_labels):
        predictions = batch_predictions.exp()[:, 1]
        return predictions, batch_labels
