import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    OUTPUT_DIR,
    RAW_SAMPLE_PATH,
    USER_PROFILE_PATH,
    AD_FEATURE_PATH,
    BEHAVIOR_LOG_PATH,
    CHUNK_SIZE,
    TRAIN_END_DATE,
)


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def run_aggregate() -> None:
    out_path = OUTPUT_DIR / "user_behavior_features.parquet"
    if out_path.exists():
        log("user_behavior_features.parquet already exists, skipping")
        return

    log("=== STEP 1: AGGREGATE ===")
    chunks = []
    total_chunks = 0

    for chunk in pd.read_csv(
        BEHAVIOR_LOG_PATH,
        chunksize=CHUNK_SIZE,
        usecols=["user", "time_stamp", "btag", "cate"],
    ):
        total_chunks += 1
        log(f"  Loading chunk {total_chunks}... ({len(chunk):,} rows)")
        chunk = chunk.dropna(subset=["user", "cate", "btag"])
        chunk["user"] = chunk["user"].astype("int32")
        chunk["cate"] = chunk["cate"].astype("int32")
        chunks.append(chunk)
        del chunk
        gc.collect()

    log("Concatenating all chunks...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    log(f"Total rows: {len(df):,}")

    log("Aggregating user-category behaviors...")
    records = []
    total_users = df["user"].nunique()
    processed = 0

    for uid, udf in df.groupby("user"):
        cate_dict = {}
        for row in udf.itertuples(index=False):
            cate = row.cate
            btag = row.btag
            if cate not in cate_dict:
                cate_dict[cate] = {"pv": 0, "fav": 0, "cart": 0, "buy": 0}
            if btag in cate_dict[cate]:
                cate_dict[cate][btag] += 1

        total_pv = sum(v["pv"] for v in cate_dict.values())
        total_fav = sum(v["fav"] for v in cate_dict.values())
        total_cart = sum(v["cart"] for v in cate_dict.values())
        total_buy = sum(v["buy"] for v in cate_dict.values())
        total_actions = total_pv + total_fav + total_cart + total_buy

        cate_scores = {
            c: (v["pv"] * 1 + v["fav"] * 2 + v["cart"] * 3 + v["buy"] * 4)
            for c, v in cate_dict.items()
        }
        top_cates = sorted(cate_scores, key=cate_scores.get, reverse=True)[:3]

        def get(i, btag):
            if len(top_cates) > i:
                return cate_dict.get(top_cates[i], {}).get(btag, 0)
            return 0

        records.append(
            {
                "user": uid,
                "total_pv": total_pv,
                "total_fav": total_fav,
                "total_cart": total_cart,
                "total_buy": total_buy,
                "total_actions": total_actions,
                "buy_rate": total_buy / max(total_actions, 1),
                "cart_rate": total_cart / max(total_actions, 1),
                "top_cate1": top_cates[0] if len(top_cates) > 0 else 0,
                "top_cate2": top_cates[1] if len(top_cates) > 1 else 0,
                "top_cate3": top_cates[2] if len(top_cates) > 2 else 0,
                "top_cate1_pv": get(0, "pv"),
                "top_cate1_buy": get(0, "buy"),
                "top_cate1_cart": get(0, "cart"),
                "top_cate2_pv": get(1, "pv"),
                "top_cate2_buy": get(1, "buy"),
                "top_cate3_pv": get(2, "pv"),
            }
        )

        processed += 1
        if processed % 100_000 == 0:
            log(f"  Progress: {processed:,} / {total_users:,} users")

    user_features = pd.DataFrame(records)

    user_features["action_group"] = pd.cut(
        user_features["total_actions"],
        bins=[0, 22, 110, float("inf")],
        labels=["sparse", "medium", "dense"],
    )

    print("\n[User counts by group]")
    print(user_features["action_group"].value_counts())
    print("\n[Action distribution by group]")
    print(
        user_features.groupby("action_group", observed=True)["total_actions"].describe()
    )

    user_features.to_parquet(out_path, index=False)
    log(f"Saved: {out_path} ({len(user_features):,} users)")
    del df, records
    gc.collect()


def run_ad_embed(batch_size: int = 256) -> None:
    out_path = OUTPUT_DIR / "ad_embeddings_reduced.npy"
    if out_path.exists():
        log("ad_embeddings_reduced.npy already exists, skipping")
        return

    log("=== STEP 2: AD EMBEDDING ===")
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    af = pd.read_csv(AD_FEATURE_PATH)
    af["brand"] = af["brand"].fillna(0).astype(int)
    af["price"] = af["price"].fillna(0)

    af["ad_text"] = af.apply(
        lambda row: (
            f"Ad in category {int(row['cate_id'])} "
            f"under campaign {int(row['campaign_id'])}. "
            f"Brand: {int(row['brand'])}. "
            f"Price: {float(row['price']):.1f}."
        ),
        axis=1,
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    ad_emb = model.encode(
        af["ad_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(OUTPUT_DIR / "ad_embeddings_adgroup_ids.npy", af["adgroup_id"].values)

    pca = PCA(n_components=64, random_state=42)
    pca.fit(ad_emb)
    ad_reduced = pca.transform(ad_emb)
    np.save(out_path, ad_reduced)
    log(f"Saved ad embeddings: {ad_reduced.shape}")
    del ad_emb, ad_reduced
    gc.collect()


def compute_intent_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pv_to_cart_rate"] = df["total_cart"] / df["total_pv"].clip(lower=1)
    df["cart_to_buy_rate"] = df["total_buy"] / df["total_cart"].clip(lower=1)
    df["pv_to_buy_rate"] = df["total_buy"] / df["total_pv"].clip(lower=1)
    df["fav_rate"] = df["total_fav"] / df["total_actions"].clip(lower=1)
    df["cate_diversity"] = (
        (df["top_cate1"] > 0).astype(int)
        + (df["top_cate2"] > 0).astype(int)
        + (df["top_cate3"] > 0).astype(int)
    )
    df["cate_concentration"] = df["top_cate1_pv"] / df["total_pv"].clip(lower=1)
    return df


def build_intent_template(row) -> str:
    pv_to_cart = row["pv_to_cart_rate"]
    cart_to_buy = row["cart_to_buy_rate"]
    fav_rate = row["fav_rate"]

    if cart_to_buy > 0.3:
        funnel_stage = "high conversion buyer"
    elif pv_to_cart > 0.05:
        funnel_stage = "active consideration shopper"
    elif fav_rate > 0.05:
        funnel_stage = "wishlist collector"
    elif row["buy_rate"] > 0:
        funnel_stage = "occasional buyer"
    else:
        funnel_stage = "browser only"

    concentration = row["cate_concentration"]
    if concentration > 0.7:
        focus = "highly focused on one category"
    elif concentration > 0.4:
        focus = "moderately focused"
    else:
        focus = "browsing across multiple categories"

    if row["total_actions"] > 100:
        activity = "highly active"
    elif row["total_actions"] > 30:
        activity = "moderately active"
    else:
        activity = "low activity"

    return (
        f"User is a {funnel_stage} with {activity} engagement. "
        f"Purchase funnel: views {int(row['total_pv'])} items, "
        f"adds {int(row['total_cart'])} to cart "
        f"(view-to-cart: {pv_to_cart:.3f}), "
        f"completes {int(row['total_buy'])} purchases "
        f"(cart-to-buy: {cart_to_buy:.3f}). "
        f"Saves {int(row['total_fav'])} items to favorites "
        f"(fav rate: {fav_rate:.3f}). "
        f"Shopping pattern: {focus} "
        f"across {int(row['cate_diversity'])} categories. "
        f"Top category engagement: "
        f"viewed {int(row['top_cate1_pv'])} times, "
        f"purchased {int(row['top_cate1_buy'])} times. "
        f"Shopping level: {int(row['shopping_level'])}. "
        f"Age group: {int(row['age_level'])}."
    )


def run_intent_embed(batch_size: int = 256, chunk_size: int = 50_000) -> None:
    out_path = OUTPUT_DIR / "intent_embeddings_reduced.npy"
    if out_path.exists():
        log("intent_embeddings_reduced.npy already exists, skipping")
        return

    log("=== STEP 3: INTENT EMBEDDING ===")
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    feat = pd.read_parquet(OUTPUT_DIR / "user_behavior_features.parquet")

    up = pd.read_csv(USER_PROFILE_PATH)
    up.columns = up.columns.str.strip()
    up["pvalue_level"] = up["pvalue_level"].fillna(0).astype(int)
    up["new_user_class_level"] = up["new_user_class_level"].fillna(
        up["new_user_class_level"].mode()[0]
    ).astype(int)

    feat = feat.merge(up, left_on="user", right_on="userid", how="left")
    feat["shopping_level"] = feat["shopping_level"].fillna(0).astype(int)
    feat["age_level"] = feat["age_level"].fillna(0).astype(int)
    del up
    gc.collect()

    feat = compute_intent_features(feat)

    log("Generating intent templates...")
    feat["intent_text"] = feat.apply(build_intent_template, axis=1)

    print("\n[Sample intent templates]")
    for g in ["sparse", "medium", "dense"]:
        mask = feat["action_group"].astype(str) == g
        if mask.sum() > 0:
            print(f"\n[{g}]\n  {feat[mask]['intent_text'].iloc[0]}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dir = OUTPUT_DIR / "intent_chunks"
    embed_dir.mkdir(exist_ok=True)

    total = len(feat)
    n_chunks = (total + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        chunk_path = embed_dir / f"chunk_{i:04d}.npy"
        if chunk_path.exists():
            log(f"  Chunk {i+1}/{n_chunks} skipped")
            continue

        start = i * chunk_size
        end = min(start + chunk_size, total)
        texts = feat["intent_text"].iloc[start:end].tolist()
        log(f"  Chunk {i+1}/{n_chunks}: {start:,}~{end:,}")

        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(chunk_path, emb)
        del emb, texts
        gc.collect()

    log("Merging chunks...")
    full_embeds = np.vstack(
        [np.load(embed_dir / f"chunk_{i:04d}.npy") for i in range(n_chunks)]
    )

    idx = np.random.choice(len(full_embeds), min(100_000, len(full_embeds)), replace=False)
    pca = PCA(n_components=64, random_state=42)
    pca.fit(full_embeds[idx])
    log(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    reduced = np.vstack(
        [pca.transform(full_embeds[i:i + chunk_size]) for i in range(0, len(full_embeds), chunk_size)]
    )

    np.save(out_path, reduced)
    np.save(OUTPUT_DIR / "intent_embeddings_user_ids.npy", feat["user"].values)

    feat.to_parquet(OUTPUT_DIR / "user_behavior_features_v2.parquet", index=False)
    log(f"Saved intent embeddings: {reduced.shape}")
    del full_embeds, reduced
    gc.collect()


SPARSE_FEATURES = [
    "user",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "brand",
    "customer",
    "shopping_level",
    "age_level",
    "final_gender_code",
    "pvalue_level",
    "new_user_class_level",
    "cms_group_id",
    "top_cate1",
    "top_cate2",
    "top_cate3",
]

DENSE_FEATURES = [
    "price",
    "buy_rate",
    "cart_rate",
    "total_pv",
    "total_fav",
    "total_cart",
    "total_buy",
    "total_actions",
    "top_cate1_pv",
    "top_cate1_buy",
    "top_cate1_cart",
    "top_cate2_pv",
    "top_cate2_buy",
    "top_cate3_pv",
    "pv_to_cart_rate",
    "cart_to_buy_rate",
    "pv_to_buy_rate",
    "fav_rate",
    "cate_diversity",
    "cate_concentration",
]


def prepare_data():
    log("=== STEP 4: PREPARE DATA ===")

    rs = pd.read_csv(RAW_SAMPLE_PATH)
    rs["time_stamp"] = pd.to_datetime(rs["time_stamp"], unit="s")
    split = pd.Timestamp(TRAIN_END_DATE)

    train_rs = rs[rs["time_stamp"] < split].copy()
    test_rs = rs[rs["time_stamp"] >= split].copy()
    del rs
    gc.collect()

    af = pd.read_csv(AD_FEATURE_PATH)
    af["brand"] = af["brand"].fillna(0).astype(int)
    af["price"] = af["price"].fillna(0).astype(float)

    train = train_rs.merge(af, on="adgroup_id", how="left")
    test = test_rs.merge(af, on="adgroup_id", how="left")
    del train_rs, test_rs, af
    gc.collect()

    uf = pd.read_parquet(OUTPUT_DIR / "user_behavior_features_v2.parquet")
    uf["action_group"] = uf["action_group"].astype(str)
    train = train.merge(uf, on="user", how="left")
    test = test.merge(uf, on="user", how="left")
    del uf
    gc.collect()

    up = pd.read_csv(USER_PROFILE_PATH)
    up.columns = up.columns.str.strip()
    up["pvalue_level"] = up["pvalue_level"].fillna(0).astype(int)
    up["new_user_class_level"] = up["new_user_class_level"].fillna(
        up["new_user_class_level"].mode()[0]
    ).astype(int)

    train = train.merge(up, left_on="user", right_on="userid", how="left")
    test = test.merge(up, left_on="user", right_on="userid", how="left")
    del up
    gc.collect()

    int_cols = [
        "cate_id",
        "campaign_id",
        "brand",
        "customer",
        "total_pv",
        "total_fav",
        "total_cart",
        "total_buy",
        "total_actions",
        "top_cate1",
        "top_cate2",
        "top_cate3",
        "shopping_level",
        "age_level",
        "final_gender_code",
        "cms_segid",
        "cms_group_id",
        "occupation",
        "new_user_class_level",
        "pvalue_level",
        "cate_diversity",
    ]

    float_cols = [
        "price",
        "buy_rate",
        "cart_rate",
        "top_cate1_pv",
        "top_cate1_buy",
        "top_cate1_cart",
        "top_cate2_pv",
        "top_cate2_buy",
        "top_cate3_pv",
        "pv_to_cart_rate",
        "cart_to_buy_rate",
        "pv_to_buy_rate",
        "fav_rate",
        "cate_concentration",
    ]

    for col in int_cols:
        if col in train.columns:
            train[col] = train[col].fillna(0).astype(int)
            test[col] = test[col].fillna(0).astype(int)

    for col in float_cols:
        if col in train.columns:
            train[col] = train[col].fillna(0).astype(float)
            test[col] = test[col].fillna(0).astype(float)

    price_cap = train["price"].quantile(0.99)
    train["price"] = train["price"].clip(upper=price_cap)
    test["price"] = test["price"].clip(upper=price_cap)

    scaler = StandardScaler()
    train[DENSE_FEATURES] = scaler.fit_transform(train[DENSE_FEATURES])
    test[DENSE_FEATURES] = scaler.transform(test[DENSE_FEATURES])

    log(f"train: {len(train):,} / test: {len(test):,}")
    return train, test
