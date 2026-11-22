# Results summary

> All numbers come from `python -m src.train` on **synthetic data** (see README). They are not real-world performance.

## categorizer

Primary metric: **macro_f1**

| model                     |   cv_macro_f1 |   cv_accuracy |   cv_weighted_f1 |   cv_macro_f1_std |   train_seconds |
|:--------------------------|--------------:|--------------:|-----------------:|------------------:|----------------:|
| Baseline: majority class  |        0.0201 |        0.1777 |           0.0536 |            0      |             1.9 |
| NaiveBayes                |        0.9615 |        0.9543 |           0.9573 |            0.0028 |             0.8 |
| NaiveBayes + SMOTE        |        0.9595 |        0.9541 |           0.957  |            0.0032 |             1.7 |
| LogReg                    |        0.9732 |        0.9633 |           0.9636 |            0.0018 |             5.7 |
| LogReg + balanced         |        0.9597 |        0.9583 |           0.959  |            0.0032 |             5.4 |
| LogReg + SMOTE            |        0.9615 |        0.959  |           0.9596 |            0.0024 |             9   |
| LinearSVM(cal)            |        0.9729 |        0.9623 |           0.9625 |            0.0019 |             5.4 |
| LinearSVM(cal) + balanced |        0.971  |        0.9614 |           0.9618 |            0.0018 |             5.7 |
| LinearSVM(cal) + SMOTE    |        0.9622 |        0.9588 |           0.9594 |            0.0022 |            13.5 |
| RandomForest + balanced   |        0.9637 |        0.9604 |           0.9605 |            0.0016 |            19.3 |
| LightGBM + balanced       |        0.9664 |        0.9614 |           0.9615 |            0.0018 |            19.2 |
| LinearSVM(cal) (tuned)    |        0.9729 |      nan      |         nan      |          nan      |           nan   |

**Winner:** LinearSVM(cal) (v2, `categorizer_v2.joblib`)  
**Test metrics:** macro_f1=0.9759, accuracy=0.9671, weighted_f1=0.9673, unseen_merchant_macro_f1=0.4143, external_hf_seen_macro_f1=1.0000, external_hf_seen_accuracy=1.0000, external_hf_unseen_macro_f1=0.5404, external_hf_unseen_accuracy=0.4353, external_hf_synthetic_only_seen_macro_f1=1.0000, external_hf_synthetic_only_seen_accuracy=1.0000, external_hf_synthetic_only_unseen_macro_f1=0.5342, external_hf_synthetic_only_unseen_accuracy=0.4267  
**Baseline (test):** macro_f1=0.0202, accuracy=0.1781

## subscription

Primary metric: **f1**

| model                                        |   cv_f1 |   cv_f1_std |   cv_precision |   cv_recall |   cv_roc_auc |   cv_pr_auc |   train_seconds |
|:---------------------------------------------|--------:|------------:|---------------:|------------:|-------------:|------------:|----------------:|
| Baseline: rule (>=3 payments & gap std < 3d) |  0.4015 |      0.0348 |       nan      |    nan      |     nan      |    nan      |           nan   |
| LogReg                                       |  0.9532 |      0.0047 |         0.9564 |      0.9501 |       0.9993 |      0.992  |             0.2 |
| LogReg + balanced                            |  0.8938 |      0.0074 |         0.8129 |      0.9927 |       0.9991 |      0.9896 |             0.2 |
| LogReg + SMOTE                               |  0.8997 |      0.0069 |         0.8226 |      0.9927 |       0.9991 |      0.9895 |             0.3 |
| RandomForest                                 |  0.9921 |      0.0032 |         0.9973 |      0.9869 |       1      |      0.9998 |             3.6 |
| RandomForest + balanced                      |  0.9921 |      0.002  |         0.9908 |      0.9934 |       1      |      0.9997 |             3.3 |
| RandomForest + SMOTE                         |  0.9931 |      0.0024 |         0.9915 |      0.9947 |       1      |      0.9998 |             9.6 |
| GradientBoosting                             |  0.9911 |      0.0037 |         0.9967 |      0.9855 |       1      |      0.9994 |             7   |
| GradientBoosting + SMOTE                     |  0.9908 |      0.0029 |         0.9869 |      0.9947 |       1      |      0.9996 |            13.8 |
| XGBoost                                      |  0.9937 |      0.0019 |         0.998  |      0.9895 |       1      |      0.9997 |             0.5 |
| XGBoost + scale_pos_weight                   |  0.9928 |      0.0025 |         0.9927 |      0.9928 |       1      |      0.9998 |             0.5 |
| XGBoost + SMOTE                              |  0.9941 |      0.0031 |         0.9934 |      0.9947 |       1      |      0.9997 |             0.6 |
| XGBoost (tuned)                              |  0.9944 |    nan      |       nan      |    nan      |     nan      |    nan      |           nan   |

**Winner:** XGBoost (v2, `subscription_v2.joblib`)  
**Test metrics:** f1=0.9918, precision=1.0000, recall=0.9837, roc_auc=0.9997, pr_auc=0.9983  
**Baseline (test):** f1=0.4097, precision=0.8080, recall=0.2745

## forecaster

Primary metric: **rmse**

| model                  |   val_rmse |   val_mae |   val_smape |   val_smape_nonzero |   val_r2 |   train_seconds |
|:-----------------------|-----------:|----------:|------------:|--------------------:|---------:|----------------:|
| Baseline: last month   |    5347.24 |  1127.11  |     22.2038 |             22.6969 |   0.5265 |           nan   |
| Baseline: 3-month mean |    4212.75 |  1067.01  |     22.9573 |             21.8233 |   0.7061 |           nan   |
| Ridge                  |    3900.07 |  1094.63  |     44.0278 |             30.7376 |   0.7481 |             0   |
| RandomForest           |    3813.05 |   959.177 |     30.0256 |             24.3532 |   0.7593 |            26.9 |
| GradientBoosting       |    3899.37 |   967.299 |     38.7334 |             25.6137 |   0.7482 |            30.6 |
| XGBoost                |    3897.33 |   992.609 |     36.1118 |             25.6144 |   0.7485 |             0.6 |
| LightGBM               |    3822.21 |   978.144 |     39.2624 |             25.6458 |   0.7581 |             0.6 |
| LightGBM (tuned)       |    3803.94 |   nan     |    nan      |            nan      | nan      |           nan   |

**Winner:** LightGBM (v2, `forecaster_v2.joblib`)  
**Test metrics:** rmse=3918.2230, mae=986.0795, smape=41.6313, smape_nonzero=25.6989, r2=0.7497  
**Baseline (test):** rmse=4460.9021, mae=1112.5688, smape=22.7110, smape_nonzero=21.9630, r2=0.6755

## persona

Primary metric: **silhouette**

| model               |   k |   silhouette |   davies_bouldin |   min_cluster_share |   inertia |   ari_vs_archetype |   train_seconds |
|:--------------------|----:|-------------:|-----------------:|--------------------:|----------:|-------------------:|----------------:|
| KMeans              |   5 |       0.1843 |           1.8148 |              0.1708 |   2917.8  |             0.7394 |            0.03 |
| GaussianMixture     |   5 |       0.1762 |           1.8265 |              0.1333 |    nan    |             0.7132 |            0.01 |
| KMeans              |   7 |       0.1721 |           1.8405 |              0.0875 |   2626.07 |             0.707  |            0.03 |
| KMeans              |   6 |       0.171  |           1.7443 |              0.0833 |   2773.67 |             0.716  |            0.03 |
| Agglomerative(Ward) |   5 |       0.1598 |           1.863  |              0.1042 |    nan    |             0.6811 |            0    |
| KMeans              |   4 |       0.1593 |           1.8079 |              0.1833 |   3258.14 |             0.5908 |            0.02 |
| Agglomerative(Ward) |   6 |       0.1578 |           1.9702 |              0.1042 |    nan    |             0.6743 |            0    |
| GaussianMixture     |   6 |       0.1572 |           1.8068 |              0.1    |    nan    |             0.6769 |            0.01 |
| KMeans              |   8 |       0.1541 |           1.7961 |              0.0542 |   2522.88 |             0.6813 |            0.03 |
| GaussianMixture     |   4 |       0.152  |           2.1193 |              0.1792 |    nan    |             0.6227 |            0.01 |

**Winner:** KMeans (k=5) (v2, `persona_v2.joblib`)  
**Test metrics:** silhouette=0.1699, davies_bouldin=1.6472, ari_vs_archetype=0.6001  
**Baseline (test):** silhouette=0.0000, note: random labels ~ 0
