# -*- coding: utf-8 -*-
"""
:File: retrain.py
:Author: Donglai Zhou
:Email: zhoudl@mail.ustc.edu.cn
"""
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = 'experimental.xlsx'
x = pd.read_excel(data, sheet_name='metals')
y = pd.read_excel(data, sheet_name='overpotential')

premodel = tf.keras.models.load_model('pre-model')
xx = premodel.predict(x)
x['G_OH'] = xx[:, 0]
x['G_O - G_OH'] = xx[:, 1]
x['delta_e'] = xx[:, 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
norm = StandardScaler().fit(y_train)
y_train_ = norm.transform(y_train)
y_test_ = norm.transform(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
model.fit(
    x_train, y_train_,
    batch_size=2, epochs=1000,
    validation_data=(x_test, y_test_), verbose=2, callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, verbose=1, min_delta=1e-5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=15)
    ]
)
model.save('model')
joblib.dump(norm, 'model/norm.pkl')
