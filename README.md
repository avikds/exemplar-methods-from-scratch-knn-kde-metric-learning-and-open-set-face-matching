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

---

Built on Deep-ML.
