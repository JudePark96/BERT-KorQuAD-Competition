import torch
import h5py
import numpy as np

from torch.utils.data import Dataset


class BertPostTrainingDataset(Dataset):
    def __init__(self, corpus_file: str, max_seq_length: int, split: str = ''):
        super(BertPostTrainingDataset, self).__init__()
        self.corpus_file = corpus_file
        self.max_seq_length = max_seq_length
        self.split = split

        with h5py.File(self.corpus_file, 'r') as features_hdf:
            self.feature_keys = list(features_hdf.keys())
            self.num_instances = np.array(features_hdf.get('next_sentence_labels')).shape[0]
        print('total %s examples : %d' % (split, self.num_instances))

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):
        features = self._read_hdf_features(index)
        anno_masked_lm_labels = self._anno_mask_inputs(features['masked_lm_ids'], features['masked_lm_positions'],
                                                       self.max_seq_length)
        curr_features = dict()
        for feat_key in features.keys():
            curr_features[feat_key] = torch.tensor(features[feat_key]).long()
        curr_features['masked_lm_labels'] = torch.tensor(anno_masked_lm_labels).long()
        return curr_features

    def _read_hdf_features(self, index):
        features = {}
        with h5py.File(self.corpus_file, 'r') as features_hdf:
            for f_key in self.feature_keys:
                features[f_key] = features_hdf[f_key][index]

        return features

    def _anno_mask_inputs(self, masked_lm_ids, masked_lm_positions, max_seq_len=512):
        anno_masked_lm_labels = [-1] * max_seq_len

        for pos, label in zip(masked_lm_positions, masked_lm_ids):
            if pos == 0: continue
            anno_masked_lm_labels[pos] = label

        return anno_masked_lm_labels
