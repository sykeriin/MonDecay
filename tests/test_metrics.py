import numpy as np

from mondecay.metrics.bootstrap import cluster_bootstrap
from mondecay.metrics.core import auroc, recall_at_fpr, threshold_at_fpr, verbalisation_rate


def test_threshold_and_recall_hand_computed():
    neg = np.arange(20) / 20.0          # 0.00 .. 0.95
    # fpr=0.05 -> may flag 1 of 20 negatives -> threshold = 2nd-highest negative = 0.90
    assert abs(threshold_at_fpr(neg, 0.05) - 0.90) < 1e-9
    scores = np.concatenate([neg, [0.95, 0.5, 0.91, 0.99]])
    labels = np.array([0] * 20 + [1] * 4, bool)
    # positives strictly above 0.90: 0.95, 0.91, 0.99 -> 3/4; realised FPR = 1/20 (only the 0.95 negative)
    assert recall_at_fpr(scores, labels, 0.05) == 0.75


def test_ties_do_not_exceed_fpr():
    neg = np.zeros(20)                  # all tied
    assert threshold_at_fpr(neg, 0.05) == 0.0
    scores = np.concatenate([neg, np.zeros(5)])
    labels = np.array([0] * 20 + [1] * 5, bool)
    assert recall_at_fpr(scores, labels, 0.05) == 0.0   # nothing strictly above the tied threshold


def test_auroc_and_degenerate():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))
    assert np.isnan(recall_at_fpr([0.1, 0.2], [0, 0]))


def test_verbalisation():
    assert verbalisation_rate([1, 0, 1, 1], [1, 1, 0, 0]) == 0.5
    assert np.isnan(verbalisation_rate([1], [0]))


def test_cluster_bootstrap_resamples_clusters_and_covers_truth():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(60), 4)                      # 60 episodes x 4 samples
    y = (rng.random(60)[cl] < 0.7).astype(float)          # outcome constant within cluster
    pt, lo, hi = cluster_bootstrap(cl, lambda idx: y[idx].mean(), n_boot=500, seed=1)
    assert lo <= pt <= hi and hi - lo > 0.05              # clustered CI is wider than a naive per-row one would be
