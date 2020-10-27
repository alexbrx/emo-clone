import os
import torch
import numpy as np
from torch.utils import data
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import copy
import sys
sys.path.insert(1, '/vol/bitbucket/apg416/project/SpeechSplit')
from hparams import hparams


class IEMOCAP(data.Dataset):
    """Dataset class for the IEMOCAP dataset."""

    def __init__(self, selected_emos, mode, meta=None):
        """Initialize and preprocess the IEMOCAP dataset."""

        self.selected_emos = selected_emos
        self.mode = mode

        self.emo2idx = {emo:idx for idx,emo in enumerate(self.selected_emos)}
        self.idx2emo = {idx:emo for idx,emo in enumerate(self.selected_emos)}

        self.meta_train = None
        self.meta_val = None
        self.meta_test = None

        self.class_weight = None

        if meta is None:
            self.meta = pd.read_csv('/vol/bitbucket/apg416/project/SpeechSplit/assets/iemocap_meta.csv')
        else:
            self.meta = copy.deepcopy(meta)

        self.preprocess()

    def preprocess(self):
        """Preprocess the IEMOCAP evaluation file."""

        #Collect information in a DataFrame
        # if meta is None:
        #     meta = pd.read_csv('/vol/bitbucket/apg416/SpeechSplit/assets/iemocap_meta.csv')

        # #Remove very short utterances
        # meta = meta[meta['length']>=1.5]

        # #Merge excited and happy
        # assert 'exc' not in self.selected_emos
        # meta['emotion'] = meta['emotion'].replace({'exc':'hap'})

        #Retain only the chosen classes and encode them
        self.meta = self.meta.loc[self.meta['emotion'].isin(self.selected_emos)]
        # meta['emotion'] = meta['emotion'].replace(self.emo2idx)
        self.meta['emotion'].replace(self.emo2idx, inplace=True)
        #Split into train and test
        # self.meta_train, self.meta_test = train_test_split(meta, test_size=0.1, random_state=1234)
        #
        # self.meta_train.reset_index(inplace=True)
        # self.meta_test.reset_index(inplace=True)
        # meta['session'] = meta['wav_file'].apply(lambda s : s[3:5])
        self.meta['session'] = self.meta['wav_file'].apply(lambda s : s[3:5])

        self.meta_train = self.meta[(self.meta['session'] == '01') | (self.meta['session'] == '02') | (self.meta['session'] == '03')]
        self.meta_val = self.meta[self.meta['session'] == '04']
        self.meta_test = self.meta[self.meta['session'] == '05']

        self.meta_train.reset_index(inplace=True)
        self.meta_val.reset_index(inplace=True)
        self.meta_test.reset_index(inplace=True)
        self.meta.reset_index(inplace=True)

        #Compute class weightings
        self.class_weight = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(self.meta_train.emotion.values), y=self.meta_train.emotion.values))

        print('Finished preprocessing the IEMOCAP dataset...')
        print('Classes: ', self.idx2emo)

    def __getitem__(self, index):
        """Return one mel spectrogram and its corresponding emotion label."""
        # meta = self.meta_train if self.mode == 'train' else self.meta_test
        if self.mode == 'train':
            meta = self.meta_train
        elif self.mode == 'val':
            meta = self.meta_val
        elif self.mode == 'test':
            meta = self.meta_test
        elif self.mode == 'full':
            meta = self.meta



        path_spmel, path_raptf0, path_emb = meta.loc[index, ['path_spmel', 'path_raptf0', 'path_emb']]
        # spk_id = meta.loc[index, 'spk_id']
        emo = meta.loc[index, 'emotion']

        melsp = np.load(path_spmel + '.npy')
        f0_org = np.load(path_raptf0 + '.npy')
        emb_org = np.load(path_emb + '.npy')

        return melsp, emb_org, f0_org, emo

    def __len__(self):
        """Return the number of spectrograms."""
        if self.mode == 'train':
            return len(self.meta_train)
        elif self.mode == 'val':
            return len(self.meta_val)
        elif self.mode == 'test':
            return len(self.meta_test)
        else:
            return len(self.meta)


class MyCollator(object):
    def __init__(self, hparams):
        self.min_len_seq = hparams.min_len_seq
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        new_batch = []
        for token in batch:
            aa, b, c, emo = token
            if len(aa) > self.max_len_seq:
                len_crop = self.max_len_seq # 1.5s ~ 3s
                left = np.random.randint(0, len(aa)-len_crop)
                # len_crop = np.random.randint(self.min_len_seq, self.max_len_seq+1, size=2) # 1.5s ~ 3s
                # left = np.random.randint(0, len(aa)-len_crop, size=2)
                # a = aa[left[0]:left[0]+len_crop[0], :]
                # c = c[left[0]:left[0]+len_crop[0]]
                a = aa[left:left+len_crop, :]
                c = c[left:left+len_crop]
            else:
                len_crop = len(aa)
                a = aa
            a = np.clip(a, 0, 1)

            a_pad = np.pad(a, ((0,self.max_len_pad-a.shape[0]),(0,0)), 'constant')
            c_pad = np.pad(c[:,np.newaxis], ((0,self.max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)

            # new_batch.append( (a_pad, b, c_pad, len_crop[0]) )
            new_batch.append( (a_pad, b, c_pad, len_crop, emo) )

        batch = new_batch

        a, b, c, d, e = zip(*batch)
        melsp = torch.from_numpy(np.stack(a, axis=0))
        spk_emb = torch.from_numpy(np.stack(b, axis=0))
        pitch = torch.from_numpy(np.stack(c, axis=0))
        len_org = torch.from_numpy(np.stack(d, axis=0))
        emo_org = torch.from_numpy(np.stack(e, axis=0))

        return melsp, spk_emb, pitch, len_org, emo_org

def get_loader(selected_emos=['ang', 'neu', 'hap', 'sad'],
                mode='train', batch_size=16, num_workers=1, drop_last=True, meta=None):
    """Build and return a data loader."""

    dataset = IEMOCAP(selected_emos, mode, meta)

    my_collator = MyCollator(hparams)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode in ['train', 'full']),
                                  collate_fn=my_collator,
                                  num_workers=1,
                                  drop_last=drop_last)
    return data_loader
