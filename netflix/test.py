import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here
n, d = X.shape

K = 4
seed = 0
mixture, post = common.init(X, K, seed)
mix_conv, post_conv, log_lh_conv = em.run(X,mixture, post)

X_predict = em.fill_matrix(X, mix_conv)

rmse = common.rmse(X_gold, X_predict)
print(log_lh_conv)

