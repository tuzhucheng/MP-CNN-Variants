class Evaluator(object):
    """
    Evaluates performance of model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device, nonstatic_embedding=None):
        self.dataset_cls = dataset_cls
        self.model = model
        self.embedding = embedding
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.device = device
        self.nonstatic_embedding = nonstatic_embedding

    def get_sentence_embeddings(self, batch):
        sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_2).transpose(1, 2)

        sent1_nonstatic, sent2_nonstatic = None, None
        if self.nonstatic_embedding is not None:
            sent1_nonstatic = self.nonstatic_embedding(batch.sentence_1).transpose(1, 2)
            sent2_nonstatic = self.nonstatic_embedding(batch.sentence_2).transpose(1, 2)

        return sent1, sent2, sent1_nonstatic, sent2_nonstatic

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')
