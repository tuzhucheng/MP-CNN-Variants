import torch
import torch.nn.functional as F

from evaluators.evaluator import Evaluator


class TRECQAEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(TRECQAEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            output = self.model(batch.a, batch.b, batch.ext_feats)
            test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).data[0]

            true_labels.append(batch.label.data.cpu())
            predictions.append(output.data.exp())

            del output

        # predictions = torch.cat(predictions).cpu().numpy()
        # true_labels = torch.cat(true_labels).cpu().numpy()
        test_cross_entropy_loss /= len(batch.dataset.examples)

        return [test_cross_entropy_loss], ['cross entropy loss']