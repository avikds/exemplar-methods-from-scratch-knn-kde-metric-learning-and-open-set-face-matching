# Exemplar Methods from Scratch: kNN, KDE, Metric Learning and Open-Set Face Matching

Build the methods of Murphy's Probabilistic Machine Learning in NumPy, keeping the training data and deciding by similarity to it, then test each claim on measured data. Reproduce the book's U-shaped curve exactly, and check on a problem with a known answer that 1-NN really reaches a factor of 2 of the Bayes error, but only once the sample is large enough. Watch the curse of dimensionality take 5-NN from 0.95 to below 0.50 with noise columns alone. Build Parzen window density estimators and pick bandwidths by rule of thumb and by leave-one-out, derive KNN from balloon densities down to machine precision, and fix Nadaraya-Watson's boundary bias with local lines. Then learn the metric: NCA and LMNN with exact gradients, followed by contrastive, triplet, N-pairs and centroid losses on the unit sphere. On Olivetti faces, split by person, you build an open-set system that says 'I have not seen this person', tune its threshold on strangers, and run a paired bake-off over many splits to learn what actually helps (whitening with no labels at all gives most of the gain). Finally, a k-d tree shows where exact search stops pruning, and locality sensitive hashing finds nearly every neighbour while scanning under 2% of the data.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** pairwise_sq_dists
- [x] **2.** knn_proba
- [x] **3.** error_vs_k
- [x] **4.** one_nn_vs_bayes
- [x] **5.** cube_edge_length
- [x] **6.** noise_dimension_curve
- [x] **7.** kde_density
- [x] **8.** silverman_bandwidth
- [x] **9.** kde_logpdf
- [x] **10.** balloon_posterior
- [x] **11.** kde_classifier
- [x] **12.** nadaraya_watson
- [x] **13.** loo_nw_bandwidth
- [x] **14.** loess
- [x] **15.** nca_objective
- [x] **16.** lmnn_objective
- [x] **17.** fit_linear_metric
- [x] **18.** whitened_pca
- [x] **19.** contrastive_loss
- [x] **20.** mine_triplets
- [x] **21.** triplet_loss
- [x] **22.** npairs_loss
- [x] **23.** centroid_loss
- [x] **24.** train_embedding
- [x] **25.** open_set_predict
- [x] **26.** tune_threshold
- [x] **27.** face_bakeoff
- [x] **28.** online_gallery
- [x] **29.** KDTree
- [x] **30.** build_lsh

## Results

```
== Part 1. K nearest neighbours (Figures 16.1 and 16.2) ==
  Figure 16.1(a), neighbours {1, 1, 1, 0, 0}: p(y=1) = 0.6
  Figure 16.2(d)  K     [1, 2, 5, 10, 20, 50, 70, 79]
                  train [0.0, 0.1, 0.15, 0.16, 0.18, 0.18, 0.27, 0.43]
                  test  [0.294, 0.313, 0.27, 0.249, 0.239, 0.243, 0.334, 0.509]
  1-NN vs the Bayes error, delta = 1:
    D =  1: Bayes 0.159, large-N limit 0.225, bound 0.267, 1-NN at N = 100, 1000, 10000: [0.246, 0.222, 0.235]
    D =  5: Bayes 0.159, large-N limit 0.225, bound 0.267, 1-NN at N = 100, 1000, 10000: [0.223, 0.236, 0.216]
    D = 20: Bayes 0.159, large-N limit 0.225, bound 0.267, 1-NN at N = 100, 1000, 10000: [0.332, 0.276, 0.247]

== Part 2. The curse of dimensionality ==
  edge of a cube holding 10% of the data in  1 dimensions: theory 0.100, simulated 0.099
  edge of a cube holding 10% of the data in 10 dimensions: theory 0.794, simulated 0.796
  edge of a cube holding 1% of the data in 10 dimensions: theory 0.631, simulated 0.633
  5-NN accuracy with 0, 8, 32, 128, 512 noise columns (scale 2): [0.94, 0.874, 0.639, 0.476, 0.401]

== Part 3. Kernel density estimation ==
  Figure 16.9 data, boxcar   h = 1.0: peak 0.250, integral 1.000
  Figure 16.9 data, boxcar   h = 2.0: peak 0.125, integral 1.000
  Figure 16.9 data, gaussian h = 1.0: peak 0.160, integral 1.000
  Figure 16.9 data, gaussian h = 2.0: peak 0.106, integral 1.000
  mean test log density on 300 points, over 5 draws:
    Gaussian   Silverman h 0.341 -> -1.429   leave-one-out h 0.321 -> -1.430
    bimodal    Silverman h 0.731 -> -1.652   leave-one-out h 0.178 -> -1.438
    Student t3 Silverman h 0.377 -> -2.381   leave-one-out h 0.853 -> -1.952

== Part 4. From KDE back to KNN ==
  largest gap between the balloon posterior and eq 16.1: 1.1102230246251565e-16
  total mass of the balloon estimate over ranges of 10, 100, 1000, 10000: [1.31, 1.37, 1.43, 1.49]
  fixed-bandwidth KDE classifier: best test error 0.244 at h = 6.97
  best KNN test error on the same data: 0.236

== Part 5. Kernel regression ==
  the Figure 16.10 notebook grid picks h = 0.1 (its smallest value)
  a finer grid picks h = 0.07, and the MSE against the true curve is 0.00263
  eq 16.45 at h = 0.0275, sigma = 0.0: coverage of the nominal 95% interval 0.966
  eq 16.45 at h = 0.0275, sigma = 0.2: coverage of the nominal 95% interval 0.997
  eq 16.45 at h = 0.0005, sigma = 0.0: coverage of the nominal 95% interval 0.176
  eq 16.45 at h = 0.0005, sigma = 0.2: coverage of the nominal 95% interval 0.877
  bias on a straight line at x = 0, 0.5, 1:  Nadaraya-Watson [0.161, -0.003, -0.152]  LOESS [0.009, -0.004, 0.005]

== Part 6. Learning a metric, 32 noise dimensions, 5-NN test accuracy ==
  draw 0: Euclidean 0.680, NCA 0.711 (training 1-J/N 0.007), LMNN 0.906
  draw 1: Euclidean 0.710, NCA 0.804 (training 1-J/N 0.000), LMNN 0.937
  draw 2: Euclidean 0.703, NCA 0.830 (training 1-J/N 0.003), LMNN 0.928

== Part 7. Ranking losses ==
  Figure 16.6(b): triples for 8 points in 2 classes: 96 against 16 centroid distances
  eq 16.21-16.22, random unit vectors : triplet sum    36.52 <= bound   755.17, and the O(NC) form gives   755.17
  eq 16.21-16.22, corners + noise 0.3 : triplet sum  -209.93 <= bound   -63.84, and the O(NC) form gives   -63.84
  eq 16.21-16.22, corners + noise 0.02: triplet sum  -397.65 <= bound  -384.67, and the O(NC) form gives  -384.67

== Part 8. Open-set matching on Olivetti faces, split by person ==
  12 splits: 17 people train, 8 calibrate, 10 enrolled with one image each, 5 strangers. Gains are paired against whitened PCA.
  12 splits cannot rank methods that are this close. Over 80 splits, the whitening step gives +0.12 AUC and the best metrics +0.02 to +0.03 more.
    whitened PCA         AUC 0.865 (+0.000 +- 0.000)  rank-1 0.807  balanced 0.737  at the oracle threshold 0.795
    raw pixels           AUC 0.713 (-0.152 +- 0.032)  rank-1 0.668  balanced 0.587  at the oracle threshold 0.666
    NCA                  AUC 0.833 (-0.032 +- 0.010)  rank-1 0.786  balanced 0.706  at the oracle threshold 0.774
    LMNN                 AUC 0.881 (+0.016 +- 0.013)  rank-1 0.847  balanced 0.749  at the oracle threshold 0.802
    contrastive          AUC 0.882 (+0.017 +- 0.011)  rank-1 0.827  balanced 0.711  at the oracle threshold 0.802
    triplet, all         AUC 0.884 (+0.019 +- 0.007)  rank-1 0.844  balanced 0.765  at the oracle threshold 0.815
    triplet, semi-hard   AUC 0.888 (+0.023 +- 0.009)  rank-1 0.844  balanced 0.782  at the oracle threshold 0.820
    triplet, hard        AUC 0.874 (+0.009 +- 0.011)  rank-1 0.828  balanced 0.754  at the oracle threshold 0.814
    N-pairs              AUC 0.896 (+0.031 +- 0.010)  rank-1 0.852  balanced 0.772  at the oracle threshold 0.827
    centroid             AUC 0.878 (+0.012 +- 0.014)  rank-1 0.816  balanced 0.724  at the oracle threshold 0.796
  false accept rate at a fixed threshold 0.35, by number of enrolled people: {1: 0.035, 5: 0.112, 10: 0.197, 20: 0.297}
  rank-1 accuracy by number of images enrolled per person: {1: 0.807, 2: 0.887, 3: 0.921, 5: 0.962}
  online gallery, 15 strangers, tau = 0.2: 6.0 identities created, ARI 0.224
  online gallery, 15 strangers, tau = 0.3: 13.5 identities created, ARI 0.470
  online gallery, 15 strangers, tau = 0.4: 27.0 identities created, ARI 0.691
  online gallery, 15 strangers, tau = 0.5: 39.2 identities created, ARI 0.714
  online gallery, 15 strangers, tau = 0.6: 57.5 identities created, ARI 0.539

== Part 9. Searching faster ==
  k-d tree in  2 dimensions: each query visits 0.19% of 5000 points
  k-d tree in  5 dimensions: each query visits 1.59% of 5000 points
  k-d tree in 10 dimensions: each query visits 57.93% of 5000 points
  k-d tree in 16 dimensions: each query visits 99.72% of 5000 points
  k-d tree in 20 dimensions: each query visits 100.00% of 5000 points
  exact cosine scan of 20,000 points: 1.23 ms per query
  LSH with  1 tables of 12 bits: recall@10 0.269, scanning 0.13% of the data
  LSH with  8 tables of 12 bits: recall@10 0.859, scanning 0.72% of the data
  LSH with 32 tables of 12 bits: recall@10 1.000, scanning 1.74% of the data

Done in 69s
```
