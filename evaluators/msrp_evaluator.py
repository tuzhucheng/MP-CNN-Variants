from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

from evaluators.evaluator import Evaluator


class MSRPEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            # Select embedding
            sent1, sent2, sent1_nonstatic, sent2_nonstatic = self.get_sentence_embeddings(batch)

            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw, sent1_nonstatic, sent2_nonstatic)
            test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).item()

            true_labels.append(batch.label.detach())
            predictions.append(output.exp().detach())

            del output

        test_cross_entropy_loss /= len(batch.dataset.examples)
        predictions = torch.cat(predictions)[:, 1].cpu().numpy()
        predictions = (predictions >= 0.5).astype(int)  # use 0.5 as default threshold
        true_labels = torch.cat(true_labels).cpu().numpy()
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        return [accuracy, f1, test_cross_entropy_loss], ['accuracy', 'f1', 'cross_entropy']

    def get_final_prediction_and_label(self, batch_predictions, batch_labels):
        predictions = batch_predictions.exp()[:, 1]
        return predictions, batch_labels
