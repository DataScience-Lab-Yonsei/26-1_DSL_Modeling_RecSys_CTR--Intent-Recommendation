import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
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
from src.config import OUTPUT_DIR
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

    log("Loading and scaling embeddings...")
    intent_emb = np.load(OUTPUT_DIR / "intent_embeddings_reduced.npy").astype(np.float32)
    intent_ids_arr = np.load(OUTPUT_DIR / "intent_embeddings_user_ids.npy")
    ad_emb = np.load(OUTPUT_DIR / "ad_embeddings_reduced.npy").astype(np.float32)
    ad_ids_arr = np.load(OUTPUT_DIR / "ad_embeddings_adgroup_ids.npy")

    intent_scaler = StandardScaler()
    ad_scaler = StandardScaler()

    intent_emb_scaled = intent_scaler.fit_transform(intent_emb).astype(np.float32)
    ad_emb_scaled = ad_scaler.fit_transform(ad_emb).astype(np.float32)

    user_id2idx = {int(uid): idx for idx, uid in enumerate(intent_ids_arr)}
    ad_id2idx = {int(aid): idx for idx, aid in enumerate(ad_ids_arr)}

    test_groups = test[["user", "action_group"]].copy()

    batch_size = 4096

    train_ds = CTRDataset(
        train,
        SPARSE_FEATURES,
        DENSE_FEATURES,
        intent_emb_scaled,
        ad_emb_scaled,
        user_id2idx,
        ad_id2idx,
    )
    test_ds = CTRDataset(
        test,
        SPARSE_FEATURES,
        DENSE_FEATURES,
        intent_emb_scaled,
        ad_emb_scaled,
        user_id2idx,
        ad_id2idx,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    log("=== Train Model B (Intent-enhanced) ===")
    model_b = DeepFM(
        feat_dims=feat_dims,
        emb_dim=16,
        dense_dim=len(DENSE_FEATURES),
        semantic_dim=intent_emb_scaled.shape[1] + ad_emb_scaled.shape[1],
    )
    model_b = train_model(model_b, train_loader, n_epochs=5)

    log("=== Evaluate Model B ===")
    results_b = evaluate_model(model_b, test_loader, test_groups)

    print("\n[Model B Results]")
    print(results_b)


if __name__ == "__main__":
    main()