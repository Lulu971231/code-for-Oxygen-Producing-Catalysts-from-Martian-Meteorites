# -*- coding: utf-8 -*-
"""
:File: bayes.py
:Author: Zhou Donglai
:Email: zhoudl@mail.ustc.edu.cn
"""
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

Metal = ['Al', 'Ca', 'Fe', 'Mg', 'Mn', 'Ni']
Al_list = [0.00149, 0.02, 0.0803, 2.759, 0.0779]
Ca_list = [0.0055, 0.1374, 0.7697, 1.417, 1.142]
Fe_list = [3.303, 1.379, 2.4, 0.97, 2.238]
Mg_list = [0.00155, 5.167, 1.306, 0.744, 1.438]
Mn_list = [0, 0.0288, 0.0521, 0.0132, 0.0541]
Ni_list = [0.328, 0, 0, 0, 0]

M_list = pd.DataFrame([Al_list, Ca_list, Fe_list, Mg_list, Mn_list, Ni_list],
                      columns=['A', 'B', 'C', 'D', 'E'],
                      index=Metal).T

premodel = tf.keras.models.load_model('pre-model')
model = tf.keras.models.load_model('model')
norm = joblib.load('model/norm.pkl')
abcdelist = [0.27105, 0.567, 0.5632, 0.935, 0.6885]


def model_pred(A, B, C, D):
    E = 1 - (A + B + C + D)
    outline = 0
    if E < 0.1:
        x = np.array([A, B, C, D, 0.1])
        x /= x.sum()
        outline = 0.1 - E
    else:
        x = np.array([A, B, C, D, E])
    x = (x / abcdelist) @ M_list.values
    x = x / x.sum()
    x = np.r_[x, premodel.predict(x.reshape(1, -1))[0]]
    return -norm.inverse_transform(model.predict(x.reshape(1, -1)))[0, 0] - outline


BO = BayesianOptimization(
    model_pred,
    {
        'A': (0.1, 0.6),
        'B': (0.1, 0.6),
        'C': (0.1, 0.6),
        'D': (0.1, 0.6),
    },
    bounds_transformer=SequentialDomainReductionTransformer(eta=0.99),
    random_state=1
)
BO.maximize(init_points=20, n_iter=280, kappa=2, kappa_decay=0.99)
out = list(BO.max['params'].values())
out.append(1 - sum(out))
param = np.array(out) / abcdelist
extractliquor = (param / param.sum()) * 100
quality = abcdelist * (extractliquor / 10)
out = (np.array(out) / abcdelist) @ M_list.values
out = out / out.sum()
print(''.join([f'{Metal[i]}: {out[i]}\n' for i in range(6)]))
