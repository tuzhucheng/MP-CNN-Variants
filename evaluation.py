from evaluators.sick_evaluator import SICKEvaluator
from evaluators.msrvid_evaluator import MSRVIDEvaluator


class MPCNNEvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    @staticmethod
    def get_evaluator(dataset_cls, model, data_loader, batch_size, device):
        if data_loader is None:
            return None

        if hasattr(dataset_cls, 'NAME') and dataset_cls.NAME == 'sick':
            return SICKEvaluator(dataset_cls, model, data_loader, batch_size, device)
        elif hasattr(dataset_cls, 'NAME') and dataset_cls.NAME == 'msrvid':
            return MSRVIDEvaluator(dataset_cls, model, data_loader, batch_size, device)
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_cls))
