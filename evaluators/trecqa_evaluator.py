from evaluators.qa_evaluator import QAEvaluator


class TRECQAEvaluator(QAEvaluator):

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device):
        super(TRECQAEvaluator, self).__init__(dataset_cls, model, embedding, data_loader, batch_size, device)
