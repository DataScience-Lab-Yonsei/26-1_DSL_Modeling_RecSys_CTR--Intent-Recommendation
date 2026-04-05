# CTR Prediction with Intent-Aware Recommendation System

## 1. Project Overview

This project focuses on improving Click-Through Rate (CTR) prediction in sparse and privacy-constrained environments, where traditional user tracking methods such as cookies are limited.

Conventional recommendation systems rely heavily on user interaction history, which leads to performance degradation in cold-start and sparse data scenarios.

To address this problem, we propose an intent-aware recommendation framework that combines:
- Aggregated user behavior features
- Semantic user intent embeddings generated from behavior summaries
- DeepFM-based CTR prediction model

The goal is to enhance recommendation performance by incorporating user intent, especially when behavioral data is limited.

---

## 2. Pipeline

The overall pipeline consists of the following steps:

1. Behavior Aggregation  
   User interaction logs are aggregated to create features such as:
   - total views, cart, purchases
   - buy rate, cart rate
   - category distribution and activity level

2. Ad Embedding  
   Ad features are converted into text descriptions and encoded using a Sentence Transformer.  
   The embeddings are then reduced using PCA.

3. Intent Feature Engineering  
   User behavior is transformed into high-level funnel features and converted into natural language templates describing user intent.

4. Intent Embedding  
   The generated intent text is encoded into dense vectors using a Sentence Transformer.

5. Data Preparation  
   All features (user, ad, behavior) are merged and normalized.  
   The dataset is split into training and test sets based on timestamp.

6. Model Training  
   - Model A: DeepFM using structured features only  
   - Model B: DeepFM with additional intent and ad embeddings  

7. Evaluation  
   Performance is evaluated using:
   - AUC
   - Log Loss
   - ECE (Expected Calibration Error)



---

## 3. Output (Evaluation)

Model A (Baseline)
- AUC: 0.6424  
- LogLoss: 0.1943  
- ECE: 0.0122  

Model B (Enhanced)
- AUC: 0.6592  
- LogLoss: 0.1925  
- ECE: 0.0076  

Key observations:
- Model B outperforms Model A in overall performance
- Improvement is especially significant for sparse users
- Intent embeddings help recover performance loss in limited data settings

---

## 4. Repository Structure

ctr_project/
- data/                  (dataset not included)
- output/                (generated results, ignored)
- scripts/
  - run_model_a.py       (baseline model execution)
  - run_model_b.py       (intent-enhanced model execution)
  - run_model_c.py       (placeholder for upper-bound model)
- src/
  - config.py            (configuration settings)
  - pipeline.py          (data processing pipeline)
  - dataset.py           (PyTorch dataset)
  - models.py            (DeepFM model)
  - train_eval.py        (training and evaluation logic)
- requirements.txt
- README.md

---

## 5. References

- Guo et al. (2017), DeepFM: A Factorization-Machine based Neural Network for CTR Prediction  
- Chen et al. (2019), Behavior Sequence Transformer (BST)  
- Reimers & Gurevych (2019), Sentence-BERT  
- Taobao Ad Click Dataset (Alibaba)
