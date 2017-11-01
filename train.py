from trainers.sick_trainer import SICKTrainer
from trainers.msrvid_trainer import MSRVIDTrainer


class MPCNNTrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    @staticmethod
    def get_trainer(dataset_name, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        if dataset_name == 'sick':
            return SICKTrainer(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        elif dataset_name == 'msrvid':
            return MSRVIDTrainer(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))
