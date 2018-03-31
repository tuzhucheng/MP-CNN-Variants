from evaluators.sick_evaluator import SICKEvaluator
from evaluators.msrp_evaluator import MSRPEvaluator
from evaluators.msrvid_evaluator import MSRVIDEvaluator
from evaluators.trecqa_evaluator import TRECQAEvaluator
from evaluators.wikiqa_evaluator import WikiQAEvaluator
from evaluators.sts_evaluator import STSEvaluator


class MPCNNEvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'sick': SICKEvaluator,
        'msrp': MSRPEvaluator,
        'msrvid': MSRVIDEvaluator,
        'trecqa': TRECQAEvaluator,
        'wikiqa': WikiQAEvaluator,
        'sts': STSEvaluator
    }

    @staticmethod
    def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, nonstatic_embedding):
        if data_loader is None:
            return None

        if not hasattr(dataset_cls, 'NAME'):
            raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

        if dataset_cls.NAME not in MPCNNEvaluatorFactory.evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return MPCNNEvaluatorFactory.evaluator_map[dataset_cls.NAME](
            dataset_cls, model, embedding, data_loader, batch_size, device, nonstatic_embedding
        )
