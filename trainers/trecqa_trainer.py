from trainers.qa_trainer import QATrainer


class TRECQATrainer(QATrainer):

    def __init__(self, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(TRECQATrainer, self).__init__(model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
