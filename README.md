# Benchmarking Protein Language Models Against a Complete Deep Mutational Map of LacI

This repository contains my Ph.D. contribution towards 'A complete deep mutational map of the lactose repressor and quantification of allosteric communication' (Herde, (**Selvakumar**), et al., 2026; manuscript under revision at *Nature Communications*).

In this work we evaluated zero-shot vs. fine-tuned ESM2 for **binary** and **multivariate** function prediction of the *lac* repressor (LacI). A **complete** experimental deep mutational scan of LacI and two **partial** scans of engineered LacI variants with inverted phenotype served as the ground truth. We trained top-layer (head), full, and contrastive fitness (ConFit) ESM2 fine-tuning strategies from 8M to 650M model size. Details of the tasks and training improvements are below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rselvak6/lacI-esm2/blob/main/esm2.ipynb)

---

## Task summary

Three proteins were assessed:

| Protein | Description |
|---|---|
| `WT` | Wild-type LacI (complete map) |
| `IA5`* | Engineered LacI variant with inverted phenotype (partial map) |
| `IA9`* | Engineered LacI variant with inverted phenotype (partial map) |

*see Richards et al., *ACS Synth Biol* (2017).

Two tasks were evaluated for each protein:

* **binary** — classify a mutation as functional / non-functional (metric: AUROC)
* **multivariate** — predict the quantitative repression / anti-repression strength (metric: Spearman ρ)

---

## ESM2 training approaches

Each approach is self-contained under `src/`:

| Approach | Module | What it does |
|---|---|---|
| **Zero-shot** | `src/esm_zeroshot/` | Masked-marginal mutation scoring from a frozen ESM2 — no training |
| **Head fine-tune** | `src/head_finetune/` | Freeze ESM2, train a mean-pooling / attention MLP head on embeddings |
| **Full fine-tune** | `src/full_finetune/` | Fine-tune all ESM2 weights with a position-aware regression head |
| **Contrastive (ConFit)** | `src/esm_confit/` | Full fine-tune with a Bradley–Terry contrastive ranking loss + KL regularization to a frozen reference |

The zero-shot scorer (`src/esm_confit/loss.py::compute_mutation_score`) uses the standard masked-marginal proxy: mask the mutated position, read off the log-probability of the mutant vs. wild-type token. The contrastive path reuses that score inside a pairwise ranking objective so the model is optimized to order variants by fitness rather than to fit absolute labels.

---

## Repo layout

```
src/
  constants.py            # LacI WT / IA5 / IA9 reference sequences, AA<->id maps
  preprocess.py           # enumerate mutations, build ESM2 representation dicts
  utils.py                # data IO, n-fold splitting (shared)
  esm_zeroshot/           # frozen-model masked-marginal scoring
  head_finetune/          # frozen backbone + trainable pooling head
  full_finetune/          # end-to-end ESM2 fine-tuning
    models.py             # ESM2 + position/mean-pooling/attention heads
    run.py                # optimized training loop (AMP + grad accumulation)
  esm_confit/             # contrastive Bradley-Terry fine-tuning
    loss.py               # mutation scoring, BT loss, KL regularization
    run.py                # optimized contrastive training loop
data/
  heatmaps/               # genotype-phenotype heatmap CSVs (WT / IA5 / IA9)
```

---

## Highlights

Here's a summary of the major training improvements and performance optimizations:

* **Bradley–Terry loss.** The first implementation of pairwise loss computed all O(n²) comparisons in a batch, so its magnitude tracked batch size instead of ranking quality. The fix rank-sorts by target score and caps comparisons to a fixed number of neighbors per item, using `F.logsigmoid` for numerical stability.
* **KL regularization.** The corrected regularization runs over full per-position distributions with `log_softmax`/`softmax` inputs and `batchmean` reduction — and KL growth is used to diagnose divergence away from the pretrained weights.
* **Throughput/memory:** precomputing the frozen reference model's forward pass once, mixed-precision training via `torch.amp`, gradient accumulation to decouple physical from effective batch size (with the LR scheduler's `total_steps` scaled accordingly), and gradient clipping to keep large effective-batch updates stable.
* **Hyperparameter optimization:** optimized learning rate, decay, cosine-annealing warmup, batch size, and lambda regularization for BT loss, with KL loss minimization and growth limitation.

---

## Run

```bash
pip install -r requirements.txt
```

Typical flow:

1. **Preprocess** — build the mutation set and ESM2 representation dictionaries for a chosen protein/model:

   ```python
   from src.preprocess import build_representaton_data_dict
   ```

2. **Choose an approach** and call its entry point in `src/<approach>/run.py` (`evaluate_esm_zeroshot`, `train_meanpooling_head`, `finetune_ESM2_position_head_optimized`, `finetune_confit_optimized`).

3. **Evaluate** with AUROC (binary) or Spearman ρ (score) via `torchmetrics`, using the provided n-fold splits.
