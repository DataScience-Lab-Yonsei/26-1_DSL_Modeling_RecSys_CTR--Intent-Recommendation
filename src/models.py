import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(
        self,
        feat_dims,
        emb_dim=16,
        dense_dim=0,
        semantic_dim=0,
        hidden_dims=[400, 400],
        dropout=0.3,
    ):
        super().__init__()
        self.n_sparse = len(feat_dims)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim + 1, emb_dim, padding_idx=0) for dim in feat_dims]
        )

        deep_input_dim = (self.n_sparse * emb_dim) + dense_dim + semantic_dim

        layers = []
        in_dim = deep_input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        self.deep = nn.Sequential(*layers)
        self.output = nn.Linear(self.n_sparse + 1 + hidden_dims[-1], 1)

    def forward(self, sparse, dense, user_emb=None, ad_emb=None):
        emb_list = [self.embeddings[i](sparse[:, i]) for i in range(self.n_sparse)]
        emb_stack = torch.stack(emb_list, dim=1)

        fm_first = torch.sum(emb_stack, dim=2)
        sum_sq = torch.sum(emb_stack, dim=1) ** 2
        sq_sum = torch.sum(emb_stack ** 2, dim=1)
        fm_second = 0.5 * torch.sum(sum_sq - sq_sum, dim=1, keepdim=True)

        emb_flat = emb_stack.view(emb_stack.size(0), -1)
        deep_input_list = [emb_flat, dense]

        if user_emb is not None:
            deep_input_list.append(user_emb)
        if ad_emb is not None:
            deep_input_list.append(ad_emb)

        deep_out = self.deep(torch.cat(deep_input_list, dim=1))
        out = self.output(torch.cat([fm_first, fm_second, deep_out], dim=1))

        return torch.sigmoid(out).squeeze(1)