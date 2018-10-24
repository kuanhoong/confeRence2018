## confeRence 2018
## Ensemble Machine Learning with R
## Poo Kuan Hoong
## 20th October 2018
## ADAX

library(SuperLearner)
library(KernelKnn)

## Boston Dataset
data(Boston, package = "MASS")

set.seed(1)

## List of learners
sl_lib = c("SL.xgboost", "SL.randomForest", "SL.glmnet", "SL.nnet", "SL.ksvm", "SL.kernelKnn", "SL.rpartPrune", "SL.lm", "SL.mean")

# Fit XGBoost, RF, Lasso, Neural Net, SVM, K-nearest neighbors, Decision Tree 
# OLS, and simple mean; create automatic ensemble.
result = SuperLearner(Y = Boston$medv, X = Boston[, -14], SL.library = sl_lib)

# Review performance of each algorithm and ensemble weights.
result

# Use external (aka nested) cross-validation to estimate ensemble accuracy.
# This will take a while to run.
result2 = CV.SuperLearner(Y = Boston$medv, X = Boston[, -14], SL.library = sl_lib)

# Plot performance of individual algorithms and compare to the ensemble.
plot(result2) + theme_minimal()

# Hyperparameter optimization --
# Fit elastic net with 5 different alphas: 0, 0.2, 0.4, 0.6, 0.8, 1.0.
# 0 corresponds to ridge and 1 to lasso.

enet = create.Learner("SL.glmnet", detailed_names = T,
                      tune = list(alpha = seq(0, 1, length.out = 5)))

sl_lib2 = c("SL.mean", "SL.lm", enet$names)

enet_sl = SuperLearner(Y = Boston$medv, X = Boston[, -14], SL.library = sl_lib2)

# Identify the best-performing alpha value or use the automatic ensemble.
enet_sl