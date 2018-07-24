import argparse
import copy
import logging
import os
import pprint
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dataset import MPCNNDatasetFactory
from evaluation import MPCNNEvaluatorFactory
from train import MPCNNTrainerFactory
from utils.serialization import load_checkpoint
from variants import VariantFactory


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, nonstatic_embedding):
    saved_model_evaluator = MPCNNEvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device, nonstatic_embedding)
    scores, metric_names = saved_model_evaluator.get_scores()
    logger.info('Evaluation metrics for {}'.format(split_name))
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join([split_name] + list(map(str, scores))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--arch', help='model architecture to use', default='mpcnn')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid, trecqa, wikiqa, sts, msrp]', default='sick')
    parser.add_argument('--word-vectors-dir', help='word vectors directory', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'Castor-data', 'embeddings', 'GloVe'))
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    parser.add_argument('--word-vectors-dim', type=int, default=300, help='number of dimensions of word vectors (default: 300)')
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--wide-conv', action='store_true', default=False, help='use wide convolution instead of narrow convolution (default: false)')
    parser.add_argument('--multichannel', action='store_true', default=False, help='use multichannels to enable embedding update (default: false)')
    parser.add_argument('--attention', choices=['none', 'basic', 'idf', 'modified_euclidean'], default='none', help='type of attention to use')
    parser.add_argument('--sparse-features', action='store_true', default=False, help='use sparse features (default: false)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adadelta'], help='optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.3, help='learning rate reduce factor after plateau (default: 0.3)')
    parser.add_argument('--patience', type=float, default=2, help='learning rate patience after seeing plateau (default: 2)')
    parser.add_argument('--momentum', type=float, default=0, help='momentum (default: 0)')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimizer epsilon (default: 1e-8)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--regularization', type=float, default=0.0001, help='Regularization for the optimizer (default: 0.0001)')
    parser.add_argument('--max-window-size', type=int, default=3, help='windows sizes will be [1,max_window_size] and infinity (default: 3)')
    parser.add_argument('--holistic-filters', type=int, default=300, help='number of holistic filters (default: 300)')
    parser.add_argument('--per-dim-filters', type=int, default=20, help='number of per-dimension filters (default: 20)')
    parser.add_argument('--hidden-units', type=int, default=150, help='number of hidden units in each of the two hidden layers (default: 150)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--run-label', type=str, help='label to describe run')
    parser.add_argument('--save-predictions', action='store_true', default=False, help='save predictions for debugging (default: false)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    logger = get_logger()
    logger.info(pprint.pformat(vars(args)))

    dataset_cls, embedding, train_loader, test_loader, dev_loader \
        = MPCNNDatasetFactory.get_dataset(args.dataset, args.word_vectors_dir, args.word_vectors_file, args.batch_size, args.device)

    if args.multichannel:
        nonstatic_embedding = copy.deepcopy(embedding)
        nonstatic_embedding.weight.requires_grad = True
    else:
        nonstatic_embedding = None

    model = VariantFactory.get_model(args, dataset_cls)

    model = model.to(device)
    embedding = embedding.to(device)
    if args.multichannel:
        nonstatic_embedding = nonstatic_embedding.to(device)

    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)

    train_evaluator = MPCNNEvaluatorFactory.get_evaluator(dataset_cls, model, embedding, train_loader, args.batch_size, args.device, nonstatic_embedding)
    test_evaluator = MPCNNEvaluatorFactory.get_evaluator(dataset_cls, model, embedding, test_loader, args.batch_size, args.device, nonstatic_embedding)
    dev_evaluator = MPCNNEvaluatorFactory.get_evaluator(dataset_cls, model, embedding, dev_loader, args.batch_size, args.device, nonstatic_embedding)

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'model_outfile': args.model_outfile,
        'lr_reduce_factor': args.lr_reduce_factor,
        'patience': args.patience,
        'tensorboard': args.tensorboard,
        'run_label': args.run_label,
        'logger': logger
    }
    trainer = MPCNNTrainerFactory.get_trainer(args.dataset, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator, nonstatic_embedding)

    # TODO currently saving & loading non-static embedding is not supported
    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)

        if args.multichannel:
            logger.info('Nonstatic embedding size: %s', nonstatic_embedding.weight.size())
        trainer.train(args.epochs)

    _, _, state_dict, _, _ = load_checkpoint(args.model_outfile)

    for k, tensor in state_dict.items():
        state_dict[k] = tensor.to(device)

    model.load_state_dict(state_dict)
    if dev_loader:
        evaluate_dataset('dev', dataset_cls, model, embedding, dev_loader, args.batch_size, args.device, nonstatic_embedding)
    evaluate_dataset('test', dataset_cls, model, embedding, test_loader, args.batch_size, args.device, nonstatic_embedding)

    if args.save_predictions:
        for dataset_name, loader in zip(('train', 'test', 'dev'), (train_loader, test_loader, dev_loader)):
            if loader is None:
                continue

            all_sent_ids, all_predictions, all_labels = [], [], []
            all_sentences_1, all_sentences_2 = [], []
            for batch in loader:
                sent_ids = batch.id.int().cpu().detach().numpy()

                # Select embedding
                sent1 = embedding(batch.sentence_1).transpose(1, 2)
                sent2 = embedding(batch.sentence_2).transpose(1, 2)

                predictions = model(sent1, sent2, batch.ext_feats)
                labels = batch.label
                predictions, labels = train_evaluator.get_final_prediction_and_label(predictions, labels)
                predictions = predictions.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                all_sent_ids.extend(sent_ids)
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_sentences_1.extend(batch.sentence_1_raw)
                all_sentences_2.extend(batch.sentence_2_raw)

            df = pd.DataFrame({'id': all_sent_ids, 'sentence1': all_sentences_1, 'sentence2': all_sentences_2, 'predictions': all_predictions, 'labels': all_labels})
            df.to_csv('{}_{}_predictions.csv'.format(args.dataset, dataset_name), index=False)
