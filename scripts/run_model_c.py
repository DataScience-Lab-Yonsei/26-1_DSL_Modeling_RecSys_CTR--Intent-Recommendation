from pathlib import Path

import inspect
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ================================================================
# 경로 설정 (프로젝트 루트 기준 상대경로)
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"
FUXICTR_PATH = PROJECT_ROOT / "FuxiCTR"
LOG_PATH = PROJECT_ROOT / "bst_log.txt"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "BST"

# 시퀀스 길이 (유저 행동 시퀀스 최근 N개)
SEQ_LEN = 20


# ================================================================
# STEP 1. 환경 설치 & FuxiCTR 클론
# ================================================================
def step1_install():
    print("\n[STEP 1] 환경 설치 확인")

    try:
        import fuxictr
        print(f"  ✅ fuxictr {fuxictr.__version__} — 스킵")
    except ImportError:
        print("  📦 fuxictr 설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fuxictr"], check=True)
        print("  ✅ fuxictr 설치 완료")

    bst_dir = FUXICTR_PATH / "model_zoo" / "BST"
    if not bst_dir.exists():
        print("  📦 FuxiCTR 클론 중...")
        subprocess.run(
            ["git", "clone", "https://github.com/reczoo/FuxiCTR.git", str(FUXICTR_PATH)],
            check=True,
        )
        print("  ✅ FuxiCTR 클론 완료")
    else:
        print("  ✅ FuxiCTR 존재 — 스킵")


# ================================================================
# STEP 2. 데이터 전처리
#   - behavior_log에서 유저별 클릭 시퀀스(cate_seq) 생성
#   - 정적 피처와 머징 후 Label Encoding
#   - 시간 기준 90/10 train/test 분리
# ================================================================
def step2_preprocess():
    print("\n[STEP 2] 데이터 전처리 확인")

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    train_path = DATA_PATH / "train.csv"
    test_path = DATA_PATH / "test.csv"

    # 이미 전처리된 경우 스킵
    if train_path.exists() and test_path.exists():
        sample = pd.read_csv(train_path, nrows=1)
        if "cate_seq" in sample.columns:
            seq = sample["cate_seq"].iloc[0]
            if seq != "|".join(["0"] * SEQ_LEN):
                print("  ✅ 전처리 완료 — 스킵")
                return

    print("  🔄 전처리 시작...")

    # ── 데이터 로드 ──────────────────────────
    print("  [1/4] 데이터 로딩 중...")
    raw = pd.read_csv(DATA_PATH / "raw_sample.csv")
    ad = pd.read_csv(DATA_PATH / "ad_feature.csv")
    user = pd.read_csv(DATA_PATH / "user_profile.csv")

    # user_profile 컬럼명 정리 (공백 제거 및 통일)
    user.columns = user.columns.str.strip()
    user = user.rename(columns={"userid": "user"})

    # ── 유저 행동 시퀀스 생성 ─────────────────
    # behavior_log의 btag 값: pv(페이지뷰=클릭), cart, fav, buy
    # 클릭 행동(pv)만 사용해 최근 SEQ_LEN개 cate 시퀀스 생성
    print(f"  [2/4] 행동 시퀀스 생성 중... (약 15~20분 소요)")
    chunks = []
    for chunk in pd.read_csv(DATA_PATH / "behavior_log.csv", chunksize=500_000):
        chunks.append(chunk[chunk["btag"] == "pv"][["user", "cate", "time_stamp"]])

    blog_click = pd.concat(chunks).sort_values(["user", "time_stamp"])

    seq_df = (
        blog_click.groupby("user")["cate"]
        .apply(
            lambda g: "|".join(
                ["0"] * (SEQ_LEN - len(g.tolist()[-SEQ_LEN:]))
                + [str(x) for x in g.tolist()[-SEQ_LEN:]]
            )
        )
        .reset_index()
    )
    seq_df.columns = ["user", "cate_seq"]

    # ── 피처 머징 ────────────────────────────
    print("  [3/4] 피처 머징 중...")
    df = raw.merge(ad, on="adgroup_id", how="left")
    df = df.merge(user, on="user", how="left")
    df = df.merge(seq_df, on="user", how="left")

    # 시퀀스 없는 유저 — 제로 패딩
    df["cate_seq"] = df["cate_seq"].fillna("|".join(["0"] * SEQ_LEN))
    # 소수점 제거 (float → int 변환)
    df["cate_seq"] = df["cate_seq"].apply(
        lambda x: "|".join([str(int(float(v))) for v in x.split("|")])
    )

    # ── Label Encoding ───────────────────────
    print("  [4/4] Label Encoding 중...")
    cat_cols = [
        "user",
        "adgroup_id",
        "cate_id",
        "campaign_id",
        "customer",
        "brand",
        "pid",
        "cms_segid",
        "cms_group_id",
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
    ]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 시간 기준 train(90%) / test(10%) 분리
    df = df.sort_values("time_stamp")
    split = int(len(df) * 0.9)
    df.iloc[:split].to_csv(train_path, index=False)
    df.iloc[split:].to_csv(test_path, index=False)
    print(f"  ✅ 전처리 완료 — Train: {split:,} / Test: {len(df) - split:,}")


# ================================================================
# STEP 3. sklearn 호환성 패치
#   - sklearn >= 1.2에서 log_loss의 eps 파라미터 제거됨
# ================================================================
def step3_patch_sklearn():
    print("\n[STEP 3] sklearn 호환성 패치 확인")

    import fuxictr.metrics as fuxi_metrics

    metrics_path = Path(inspect.getfile(fuxi_metrics))
    code = metrics_path.read_text(encoding="utf-8")

    if "eps=1e-7" not in code:
        print("  ✅ 이미 패치됨 — 스킵")
        return

    print("  🔧 패치 적용 중...")
    code = code.replace(
        "log_loss(y_true, y_pred, eps=1e-7)",
        "log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7))",
    )
    if "import numpy as np" not in code:
        code = "import numpy as np\n" + code

    metrics_path.write_text(code, encoding="utf-8")
    print("  ✅ sklearn 패치 완료")


# ================================================================
# STEP 4. BST.py 차원 패치
#   - FuxiCTR의 sequence 피처가 2D로 출력될 때 3D로 맞춰주는 패치
# ================================================================
def step4_patch_bst():
    print("\n[STEP 4] BST.py 차원 패치 확인")

    bst_path = FUXICTR_PATH / "model_zoo" / "BST" / "src" / "BST.py"
    code = bst_path.read_text(encoding="utf-8")

    if "sequence_emb.dim()" in code:
        print("  ✅ 이미 패치됨 — 스킵")
        return

    print("  🔧 패치 적용 중...")
    old = (
        "            target_emb = self.concat_embedding(target_field, feature_emb_dict)\n"
        "            sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)\n"
        "            concat_seq_emb = torch.cat([sequence_emb, target_emb.unsqueeze(1)], dim=1)"
    )
    new = (
        "            target_emb = self.concat_embedding(target_field, feature_emb_dict)\n"
        "            sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)\n"
        "            # 차원 통일: 2D → 3D (batch, seq_len, dim)\n"
        "            if target_emb.dim() == 2:\n"
        "                target_emb = target_emb.unsqueeze(1)\n"
        "            if sequence_emb.dim() == 2:\n"
        "                sequence_emb = sequence_emb.unsqueeze(1)\n"
        "            concat_seq_emb = torch.cat([sequence_emb, target_emb], dim=1)"
    )

    if old not in code:
        print("  ⚠️  패치 대상 코드 없음 — BST.py 버전이 다를 수 있음")
        return

    bst_path.write_text(code.replace(old, new), encoding="utf-8")
    print("  ✅ BST.py 패치 완료")


# ================================================================
# STEP 5. 학습 Config 생성
#   - dataset_config.yaml : 피처 정의
#   - model_config.yaml   : 하이퍼파라미터
# ================================================================
def step5_config():
    print("\n[STEP 5] 학습 Config 생성")

    config_dir = FUXICTR_PATH / "model_zoo" / "BST" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.mkdir(parents=True, exist_ok=True)

    dataset_config = f"""
taobao_x1:
    data_format: csv
    train_data: {train_path_str(DATA_PATH / "train.csv")}
    valid_data: {train_path_str(DATA_PATH / "test.csv")}
    test_data:  {train_path_str(DATA_PATH / "test.csv")}
    label_col: {{name: clk, dtype: float}}
    feature_cols:
        - {{name: user,                 active: True, dtype: str,   type: categorical}}
        - {{name: adgroup_id,           active: True, dtype: str,   type: categorical}}
        - {{name: pid,                  active: True, dtype: str,   type: categorical}}
        - {{name: cate_id,              active: True, dtype: str,   type: categorical}}
        - {{name: campaign_id,          active: True, dtype: str,   type: categorical}}
        - {{name: customer,             active: True, dtype: str,   type: categorical}}
        - {{name: brand,                active: True, dtype: str,   type: categorical}}
        - {{name: cms_segid,            active: True, dtype: str,   type: categorical}}
        - {{name: cms_group_id,         active: True, dtype: str,   type: categorical}}
        - {{name: final_gender_code,    active: True, dtype: str,   type: categorical}}
        - {{name: age_level,            active: True, dtype: str,   type: categorical}}
        - {{name: pvalue_level,         active: True, dtype: str,   type: categorical}}
        - {{name: shopping_level,       active: True, dtype: str,   type: categorical}}
        - {{name: occupation,           active: True, dtype: str,   type: categorical}}
        - {{name: new_user_class_level, active: True, dtype: str,   type: categorical}}
        - {{name: price,                active: True, dtype: float, type: numeric}}
        - {{name: cate_seq,             active: True, dtype: str,   type: sequence,
           splitter: "|", max_len: {SEQ_LEN}, embedding_dim: 32, feature_encoder: None}}
"""

    model_config = f"""
Base:
    model: BST
    dataset_id: taobao_x1
    loss: binary_crossentropy
    metrics: [AUC, logloss]
    task: binary_classification
    optimizer: adam
    learning_rate: 1.0e-3
    embedding_regularizer: 1.0e-6
    net_regularizer: 0
    batch_size: 4096
    embedding_dim: 32
    epochs: 10
    shuffle: True
    seed: 42
    monitor: {{AUC: 1}}
    monitor_mode: max
    patience: 2
    data_root: {train_path_str(DATA_PATH)}
    model_root: {train_path_str(CKPT_PATH)}
    num_workers: 4
    verbose: 1

BST_taobao_x1:
    model: BST
    dataset_id: taobao_x1
    # DNN 구조
    dnn_hidden_units: [256, 128, 64]
    dnn_activations: ReLU
    # Transformer 구조
    num_heads: 4
    stacked_transformer_layers: 2
    attention_dropout: 0.1
    # 정규화
    net_dropout: 0.1
    batch_norm: False
    layer_norm: True
    use_residual: True
    use_position_emb: True
    # 시퀀스 풀링 방식 (target: 타겟 아이템 위치의 출력만 사용)
    seq_pooling_type: target
    # 시퀀스 피처 매핑
    bst_target_field: cate_id
    bst_sequence_field: cate_seq
"""

    (config_dir / "dataset_config.yaml").write_text(dataset_config, encoding="utf-8")
    (config_dir / "model_config.yaml").write_text(model_config, encoding="utf-8")

    # parquet 캐시 무조건 삭제 (이전 캐시가 남아있으면 잘못된 데이터로 학습될 수 있음)
    for cache_path in [DATA_PATH / "taobao_x1", PROJECT_ROOT / "taobao_x1"]:
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print(f"  🗑️  이전 캐시 삭제: {cache_path}")

    print("  ✅ Config 생성 완료")


def train_path_str(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


# ================================================================
# STEP 6. BST 모델 학습
# ================================================================
def step6_train():
    print("\n[STEP 6] BST 학습 시작")
    print("  (Epoch마다 AUC 출력, early stopping patience=2)\n")

    log_file = open(LOG_PATH, "w", encoding="utf-8")
    result = subprocess.Popen(
        [
            sys.executable,
            "run_expid.py",
            "--config",
            "config",
            "--expid",
            "BST_taobao_x1",
            "--gpu",
            "0",
        ],
        cwd=str(FUXICTR_PATH / "model_zoo" / "BST"),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    print(f"  PID: {result.pid}\n")

    last_pos = 0
    while True:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(last_pos)
                new_content = f.read()
                last_pos = f.tell()

            if new_content:
                for line in new_content.split("\n"):
                    if any(
                        k in line
                        for k in [
                            "Epoch",
                            "AUC",
                            "best model",
                            "Early stopping",
                            "Traceback",
                            "Error",
                        ]
                    ):
                        print(" ", line)

        if result.poll() is not None:
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            print("\n  ── 최종 로그 ──────────────────────────")
            print("".join(lines[-15:]))
            break

        time.sleep(15)


# ================================================================
# STEP 7. ECE & Calibration 평가
#   - ECE (Expected Calibration Error): 예측 확률의 신뢰도 측정
#   - Reliability Diagram: 캘리브레이션 시각화
#   - 예측값 분포: 클릭/비클릭 분포 비교
# ================================================================
def step7_calibration():
    print("\n[STEP 7] ECE & Calibration 평가")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(FUXICTR_PATH))
    sys.path.insert(0, str(FUXICTR_PATH / "model_zoo" / "BST" / "src"))

    from fuxictr.utils import load_config
    from fuxictr.pytorch.dataloaders import RankDataLoader
    from fuxictr.features import FeatureMap
    from BST import BST as BSTModel

    # 테스트 정답값 로드
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    y_true = test_df["clk"].values

    # 저장된 예측값 파일 탐색
    pred_file = None
    for root, dirs, files in os.walk(CKPT_PATH):
        for f in files:
            if f.endswith("_pred.npy") or f.endswith("test_pred.npy"):
                pred_file = Path(root) / f
                break

    if pred_file is not None:
        y_pred = np.load(pred_file)
        print(f"  ✅ 예측값 로드: {pred_file}")
    else:
        # 체크포인트에서 모델 로드 후 직접 예측
        print("  🔄 모델 로드 후 예측 중...")
        config_dir = FUXICTR_PATH / "model_zoo" / "BST" / "config"
        params = load_config(str(config_dir), "BST_taobao_x1")

        feature_map = FeatureMap(params["dataset_id"], params["data_root"])
        feature_map.load(os.path.join(params["data_root"], params["dataset_id"], "feature_map.json"))

        model = BSTModel(feature_map, **params)

        ckpt_files = []
        for root, dirs, files in os.walk(CKPT_PATH):
            for f in files:
                if f.endswith(".model") or f.endswith(".pt") or f.endswith(".bin"):
                    ckpt_files.append(Path(root) / f)

        if ckpt_files:
            ckpt_files.sort(key=os.path.getmtime, reverse=True)
            model.load_weights(str(ckpt_files[0]))
            print(f"  ✅ 체크포인트 로드: {ckpt_files[0]}")

        test_gen = RankDataLoader(
            feature_map,
            params["test_data"],
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=1,
        ).make_iterator()
        y_pred = model.predict(test_gen)

    # ── ECE 계산 (10개 bin) ────────────────────
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    bin_accs, bin_confs = [], []

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (y_pred >= lower) & (y_pred < upper)
        count = mask.sum()
        if count > 0:
            acc = y_true[mask].mean()
            conf = y_pred[mask].mean()
            ece += (count / len(y_pred)) * abs(acc - conf)
        else:
            acc, conf = 0.0, (lower + upper) / 2
        bin_accs.append(acc)
        bin_confs.append(conf)

    # ── Calibration 그래프 ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BST Model Calibration Analysis", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    bin_centers = [(l + u) / 2 for l, u in zip(bin_lowers, bin_uppers)]
    ax1.bar(
        bin_centers,
        bin_accs,
        width=0.09,
        alpha=0.7,
        color="steelblue",
        label="Actual fraction",
    )
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(f"Reliability Diagram  (ECE = {ece:.4f})", fontsize=12)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.hist(
        y_pred[y_true == 0],
        bins=50,
        alpha=0.6,
        color="steelblue",
        label="Non-click (y=0)",
        density=True,
    )
    ax2.hist(
        y_pred[y_true == 1],
        bins=50,
        alpha=0.6,
        color="orange",
        label="Click (y=1)",
        density=True,
    )
    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Predicted Probability Distribution", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = DATA_PATH / "calibration_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print("\n  ══════════════════════════════════════")
    print("          BST 모델 최종 평가 요약         ")
    print("  ══════════════════════════════════════")
    print(f"  ECE (Expected Calibration Error) : {ece:.4f}")
    print(f"  실제 CTR (Positive Rate)         : {y_true.mean():.4f}")
    print(f"  예측 평균 확률                   : {y_pred.mean():.4f}")
    print(f"  테스트 샘플 수                   : {len(y_true):,}")
    print(f"  Calibration 그래프 저장 경로     : {save_path}")
    print("  ══════════════════════════════════════")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    step1_install()
    step2_preprocess()
    step3_patch_sklearn()
    step4_patch_bst()
    step5_config()
    step6_train()
    step7_calibration()
'''

out_path = Path("/mnt/data/run_model_c.py")
out_path.write_text(code, encoding="utf-8")
print(f"Saved: {out_path}")
