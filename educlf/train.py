import argparse
import json

import datasets
import torch
import numpy

from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from .dataset import load_data
from .model import IntentClassifierModel
from sentence_transformers import SentenceTransformer


def rank_list(model, reference, candidates):
    reference_repr = model.encode(reference)
    candidates_repr = [model.encode(c) for c in candidates]
    distances = [(cosine(reference_repr, cr), c) for cr, c in zip(candidates_repr, candidates)]
    return sorted(distances, key=lambda c: c[0] )


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

    if args.model in ['robeczech', 'eleczech']:
        train_dataset, dev_dataset = load_data(args.data_fn)
        all_labels = set(train_dataset['labels'])
        label_mapping = {name: n for n, name in enumerate(all_labels)}
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
    else:
        train_dataset, dev_dataset = datasets.load_dataset('banking77', split=['train', 'test'])

    clf_model = IntentClassifierModel(args.model, device, label_mapping, args.out_dir)
    clf_model.train(train_dataset, dev_dataset)
    clf_model.save()
    predictions = clf_model.predict_from_dataset(dev_dataset)
    true = [l for l in predictions.label_ids]
    pred = [p for p in numpy.argmax(predictions.predictions, axis=-1)]
    cm = confusion_matrix(true, pred)
    dsp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[inv_label_mapping[i] for i in range(len(inv_label_mapping))])
    dsp.plot(xticks_rotation='vertical')
    plt.savefig('cm.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--model', type=str, default='robeczech')
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    main(args)
