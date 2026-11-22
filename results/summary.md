# Results summary

> All numbers come from `python -m src.train` on **synthetic data** (see README). They are not real-world performance.

## categorizer

Primary metric: **macro_f1**

| model                     |   cv_macro_f1 |   cv_accuracy |   cv_weighted_f1 |   cv_macro_f1_std |   train_seconds |
|:--------------------------|--------------:|--------------:|-----------------:|------------------:|----------------:|
| Baseline: majority class  |        0.0201 |        0.1778 |           0.0537 |            0      |             2.7 |
| NaiveBayes                |        0.9621 |        0.9546 |           0.9581 |            0.0026 |             1.2 |
| NaiveBayes + SMOTE        |        0.9603 |        0.9552 |           0.958  |            0.0027 |             2.4 |
| LogReg                    |        0.9732 |        0.9648 |           0.965  |            0.003  |             4.4 |
| LogReg + balanced         |        0.9596 |        0.9607 |           0.9612 |            0.0016 |             4.2 |
| LogReg + SMOTE            |        0.9627 |        0.9617 |           0.9621 |            0.0022 |             7.1 |
| LinearSVM(cal)            |        0.9734 |        0.9649 |           0.9652 |            0.0025 |            13.1 |
| LinearSVM(cal) + balanced |        0.9716 |        0.9634 |           0.9638 |            0.0032 |            13.1 |
| LinearSVM(cal) + SMOTE    |        0.9628 |        0.9613 |           0.9617 |            0.0021 |            28.3 |
| RandomForest + balanced   |        0.962  |        0.9605 |           0.9606 |            0.0035 |            56.4 |
| LightGBM + balanced       |        0.9655 |        0.9616 |           0.9616 |            0.0028 |           174.5 |
| LogReg (tuned)            |        0.9735 |      nan      |         nan      |          nan      |           nan   |

**Winner:** LogReg (v1, `categorizer_v1.joblib`)  
**Test metrics:** macro_f1=0.9736, accuracy=0.9667, weighted_f1=0.9669, unseen_merchant_macro_f1=0.5324  
**Baseline (test):** macro_f1=0.0201, accuracy=0.1777

## subscription

Primary metric: **f1**

| model                                        |   cv_f1 |   cv_f1_std |   cv_precision |   cv_recall |   cv_roc_auc |   cv_pr_auc |   train_seconds |
|:---------------------------------------------|--------:|------------:|---------------:|------------:|-------------:|------------:|----------------:|
| Baseline: rule (>=3 payments & gap std < 3d) |  0.3724 |      0.0188 |       nan      |    nan      |     nan      |    nan      |           nan   |
| LogReg                                       |  0.953  |      0.0081 |         0.9583 |      0.9479 |       0.9993 |      0.9918 |             0.2 |
| LogReg + balanced                            |  0.8901 |      0.0088 |         0.8081 |      0.9908 |       0.9992 |      0.9893 |             0.4 |
| LogReg + SMOTE                               |  0.8949 |      0.0125 |         0.8167 |      0.9901 |       0.9992 |      0.9891 |             0.3 |
| RandomForest                                 |  0.9911 |      0.0022 |         0.9947 |      0.9875 |       1      |      0.9997 |            10.6 |
| RandomForest + balanced                      |  0.9898 |      0.002  |         0.9876 |      0.9921 |       1      |      0.9996 |             9.6 |
| RandomForest + SMOTE                         |  0.9895 |      0.0022 |         0.9864 |      0.9927 |       1      |      0.9996 |            33.1 |
| GradientBoosting                             |  0.9878 |      0.0034 |         0.9907 |      0.9848 |       0.9999 |      0.9993 |            13.9 |
| GradientBoosting + SMOTE                     |  0.9833 |      0.0044 |         0.9748 |      0.992  |       1      |      0.9993 |            26.3 |
| XGBoost                                      |  0.9904 |      0.0041 |         0.9927 |      0.9882 |       1      |      0.9996 |             1   |
| XGBoost + scale_pos_weight                   |  0.9898 |      0.0032 |         0.9869 |      0.9928 |       1      |      0.9996 |             1.3 |
| XGBoost + SMOTE                              |  0.9908 |      0.0026 |         0.9895 |      0.9921 |       1      |      0.9996 |             1.8 |
| XGBoost (tuned)                              |  0.9911 |    nan      |       nan      |    nan      |     nan      |    nan      |           nan   |

**Winner:** XGBoost (v1, `subscription_v1.joblib`)  
**Test metrics:** f1=0.9852, precision=0.9946, recall=0.9759, roc_auc=0.9999, pr_auc=0.9989  
**Baseline (test):** f1=0.4177, precision=0.8387, recall=0.2781

## forecaster

Primary metric: **rmse**

| model                  |   val_rmse |   val_mae |   val_smape |   val_smape_nonzero |   val_r2 |   train_seconds |
|:-----------------------|-----------:|----------:|------------:|--------------------:|---------:|----------------:|
| Baseline: last month   |    4504.13 |  1030.2   |     21.2747 |             21.9831 |   0.5612 |           nan   |
| Baseline: 3-month mean |    3755.37 |  1001.07  |     22.0078 |             21.5536 |   0.6949 |           nan   |
| Ridge                  |    3441.61 |  1070.42  |     46.2671 |             32.2507 |   0.7438 |             0.1 |
| RandomForest           |    3329.08 |   928.393 |     34.0954 |             24.7896 |   0.7603 |            44.4 |
| GradientBoosting       |    3385.99 |   938.096 |     41.6808 |             26.7599 |   0.752  |            55   |
| XGBoost                |    3371.36 |   989.548 |     36.6178 |             26.6169 |   0.7541 |             1.5 |
| LightGBM               |    3351.13 |   977.476 |     40.6413 |             26.9371 |   0.7571 |             1.1 |
| LightGBM (tuned)       |    3296.15 |   nan     |    nan      |            nan      | nan      |           nan   |

**Winner:** LightGBM (v1, `forecaster_v1.joblib`)  
**Test metrics:** rmse=3442.8508, mae=930.8723, smape=41.6987, smape_nonzero=25.4457, r2=0.7596  
**Baseline (test):** rmse=3880.7659, mae=1032.9755, smape=22.1835, smape_nonzero=21.7820, r2=0.6946

## persona

Primary metric: **silhouette**

| model           |   k |   silhouette |   davies_bouldin |   min_cluster_share |   inertia |   ari_vs_archetype |   train_seconds |
|:----------------|----:|-------------:|-----------------:|--------------------:|----------:|-------------------:|----------------:|
| KMeans          |   6 |       0.1884 |           1.7264 |              0.0875 |   2694.74 |             0.7031 |            0.01 |
| KMeans          |   7 |       0.1848 |           1.7035 |              0.075  |   2587.8  |             0.6644 |            0.01 |
| GaussianMixture |   7 |       0.1787 |           1.7499 |              0.0833 |    nan    |             0.667  |            0    |
| KMeans          |   5 |       0.1762 |           1.8314 |              0.1458 |   2947.99 |             0.5657 |            0.01 |
| KMeans          |   8 |       0.1757 |           1.7732 |              0.0625 |   2472.42 |             0.581  |            0.01 |
| GaussianMixture |   6 |       0.1716 |           1.8233 |              0.1333 |    nan    |             0.7095 |            0    |
| GaussianMixture |   5 |       0.1687 |           1.8906 |              0.1083 |    nan    |             0.6705 |            0    |
| KMeans          |   4 |       0.1653 |           1.9056 |              0.1917 |   3231.57 |             0.4827 |            0.01 |
| GaussianMixture |   4 |       0.1581 |           1.893  |              0.0792 |    nan    |             0.4884 |            0.01 |
| GaussianMixture |   3 |       0.1558 |           2.1303 |              0.325  |    nan    |             0.4407 |            0    |

**Winner:** KMeans (k=7) (v2, `persona_v2.joblib`)  
**Test metrics:** silhouette=0.1717, davies_bouldin=1.5159, ari_vs_archetype=0.5598  
**Baseline (test):** silhouette=0.0000, note: random labels ~ 0
