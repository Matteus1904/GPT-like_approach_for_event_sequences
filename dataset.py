import torch
import random

from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.feature_dict import FeatureDict


class MlmNoSliceDataset(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    data:
        List with all sequences
    """
    def __init__(self,
                 data,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return self.process(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.process(feature_arrays)

    def process(self, feature_arrays):
        feature_arrays = {k: v for k, v in feature_arrays.items() if FeatureDict.is_seq_feature(k, v)}
        return feature_arrays

    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch)