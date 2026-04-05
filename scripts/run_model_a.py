import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import (
    log,
    run_aggregate,
    run_ad_embed,
    run_intent_embed,
    prepare_data,
    SPARSE_FEATURES,
    DENSE_FEATURES,
)
from src.dataset import CTRDataset
from src.models import DeepFM
from src.train_eval import train_model, evaluate_model


def main():
    run_aggregate()
    run_ad_embed()
    run_intent_embed()

    train, test = prepare_data()

    feat_dims = []
    for col in SPARSE_FEATURES:
        max_val = max(train[col].max(), test[col].max())
        feat_dims.append(int(max_val))

    test_groups = test[["user", "action_group"]].copy()

    batch_size = 4096

    train_ds = CTRDataset(train, SPARSE_FEATURES, DENSE_FEATURES)
    test_ds = CTRDataset(test, SPARSE_FEATURES, DENSE_FEATURES)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    log("=== Train Model A (Baseline) ===")
    model_a = DeepFM(
        feat_dims=feat_dims,
        emb_dim=16,
        dense_dim=len(DENSE_FEATURES),
        semantic_dim=0,
    )
    model_a = train_model(model_a, train_loader, n_epochs=5)

    log("=== Evaluate Model A ===")
    results_a = evaluate_model(model_a, test_loader, test_groups)

    print("\n[Model A Results]")
    print(results_a)


if __name__ == "__main__":
    main()