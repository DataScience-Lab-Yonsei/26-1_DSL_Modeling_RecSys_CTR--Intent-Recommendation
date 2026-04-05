import numpy as np
import torch
from torch.utils.data import Dataset


class CTRDataset(Dataset):
    def __init__(
        self,
        df,
        sparse_feats,
        dense_feats,
        user_emb=None,
        ad_emb=None,
        user_id2idx=None,
        ad_id2idx=None,
    ):
        self.labels = df["clk"].values.astype(np.float32)
        self.sparse = df[sparse_feats].values.astype(np.int64)
        self.dense = df[dense_feats].values.astype(np.float32)

        self.user_emb = user_emb
        self.ad_emb = ad_emb
        self.user_ids = df["user"].values.astype(np.int64)
        self.ad_ids = df["adgroup_id"].values.astype(np.int64)
        self.user_id2idx = user_id2idx or {}
        self.ad_id2idx = ad_id2idx or {}
        self.use_semantic = (user_emb is not None and ad_emb is not None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "label": torch.tensor(self.labels[idx]),
            "sparse": torch.tensor(self.sparse[idx]),
            "dense": torch.tensor(self.dense[idx]),
        }

        if self.use_semantic:
            uid = int(self.user_ids[idx])
            aid = int(self.ad_ids[idx])
            uidx = self.user_id2idx.get(uid, -1)
            aidx = self.ad_id2idx.get(aid, -1)

            item["user_emb"] = torch.tensor(
                self.user_emb[uidx]
                if uidx >= 0
                else np.zeros(self.user_emb.shape[1], dtype=np.float32)
            )
            item["ad_emb"] = torch.tensor(
                self.ad_emb[aidx]
                if aidx >= 0
                else np.zeros(self.ad_emb.shape[1], dtype=np.float32)
            )

        return item