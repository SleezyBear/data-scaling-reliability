# Data Scaling and Model Reliability

An empirical study of whether more training data always helps. It measures how seven classifiers behave as clean training data grows, and how that changes once low-quality data is mixed in: flipped labels, noisy features, and near-duplicate "weak" samples. The question is where adding data stops paying off, and how much bad data each model can absorb before its reliability breaks down.

## Design

**Datasets** (binary classification):
- Adult Income (fetched from OpenML)
- Give Me Some Credit (Kaggle)
- Credit Card Fraud (Kaggle)

**Models:** logistic regression, Gaussian naive Bayes, decision tree, random forest, gradient boosting, k-nearest neighbours and a small MLP, with class weighting where the model supports it.

**Data quality conditions:**
- *Label noise:* a fraction of labels flipped (0%, 10%, 20%, 30%).
- *Feature noise:* Gaussian noise added to features (σ = 0, 0.02, 0.05, 0.1).
- *Weak samples:* near-duplicate rows with tiny perturbations appended (0%, 10%, 20%, 30%).

**Metrics:** accuracy, F1 and ROC-AUC, averaged over 10 repeats, with the spread across repeats used as a stability measure.

Every run is deterministic. One global seed derives separate sub-seeds for splitting, noise and model initialization, so any single condition can be reproduced on its own. Progress is checkpointed atomically, so an interrupted run resumes where it stopped.

## Phases

| Phase | What it does |
|---|---|
| 1 | Clean baseline on the full training set |
| 2 | Clean scaling: train on 20%, 40%, 60%, 80% and 100% of the data (stratified) |
| 3 | A fixed mix of all three noise types at every training size |
| 4 | Each noise type on its own, at each level |
| 5 | Noise enrichment analysis |
| 6 | Statistical tests: Wilcoxon signed-rank (clean vs noisy, scaling effects), Friedman tests with pairwise post-hoc comparisons, and Holm and Benjamini–Hochberg corrections for multiple testing |
| 7 | Figures: scaling curves, noise-response curves, relative-drop heatmaps, breakdown curves, effect-size forest plots, rank comparisons and multi-objective views |
| 8 | Threshold detection: the saturation point where more clean data stops helping, the noise level where a model's performance breaks down, and inflection points in the scaling curves |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Adult Income downloads automatically. For the other two datasets, download them from Kaggle and place them in `data/`:

- `cs-training.csv` from [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data), renamed to `give_me_some_credit.csv`
- `creditcard.csv` from [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Then run all eight phases:

```bash
cd src
python experiment.py
```

## Layout

```
src/experiment.py            Phases 1–5 and the run driver
src/data_loader.py           Dataset loading and preprocessing
src/models.py                The seven classifiers
src/noise.py                 Label noise, feature noise and weak samples
src/statistical_tests.py     Phase 6
src/visualization.py         Phase 7
src/threshold_detection.py   Phase 8
```

## License

MIT. See [`LICENSE`](LICENSE).
