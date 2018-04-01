from trainers.sick_trainer import SICKTrainer
from trainers.msrp_trainer import MSRPTrainer
from trainers.msrvid_trainer import MSRVIDTrainer
from trainers.trecqa_trainer import TRECQATrainer
from trainers.wikiqa_trainer import WikiQATrainer
from trainers.sts_trainer import STSTrainer


class MPCNNTrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'msrp': MSRPTrainer,
        'msrvid': MSRVIDTrainer,
        'trecqa': TRECQATrainer,
        'wikiqa': WikiQATrainer,
        'sts': STSTrainer
    }

    @staticmethod
    def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None, nonstatic_embedding=None):
        if dataset_name not in MPCNNTrainerFactory.trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return MPCNNTrainerFactory.trainer_map[dataset_name](
            model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator, nonstatic_embedding
        )
