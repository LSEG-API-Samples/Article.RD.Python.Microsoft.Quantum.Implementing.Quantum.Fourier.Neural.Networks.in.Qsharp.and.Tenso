# %%
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import refinitiv.data as rd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from QuantumDense import VQNNModel

RIC_LIST = ['IBM.N']

if not os.path.isfile('./data/ibm.csv'):
    rd.open_session()
    df = rd.get_history(universe=RIC_LIST,
                            fields=['TR.CLOSEPRICE'],
                            start="2016-01-01",
                            end="2022-01-01").dropna()
    df.to_csv('./data/ibm.csv')
else:
    df = pd.read_csv('./data/ibm.csv')

df.rename(columns={'Close Price': 'CLOSE'}, inplace=True)
df['RETURNS'] = df['CLOSE'].pct_change()
df['PRX_MA_ND'] = df['CLOSE'].rolling(window=5).mean()
df['VOLATILITY'] = df['CLOSE'].rolling(window=5).std()
df['TP1_RETURNS'] = df['RETURNS'].shift(-1)

df.dropna(inplace=True)
df = df.set_index('Date')

df_x = df[['RETURNS', 'PRX_MA_ND', 'VOLATILITY']]
df_y = df['TP1_RETURNS']

df_x_scaler = MinMaxScaler().fit(df_x)

forward_test_date = '2021-03-01'

fdf_x = df_x.loc[forward_test_date:]
fdf_y = df_y.loc[forward_test_date:]
df_x = df_x.loc['2021-01-01':'2021-03-01']
df_y = df_y.loc['2021-01-01':'2021-03-01']

fdf_prx = df.loc[forward_test_date:]['CLOSE']
fdf_y_len = len(fdf_y)

df_x_scaled = pd.DataFrame(df_x_scaler.transform(df_x))
fdf_x_scaled = pd.DataFrame(df_x_scaler.transform(fdf_x))

x_train, x_test, y_train, y_test = train_test_split(df_x_scaled,
                                                    df_y,
                                                    test_size=0.25,
                                                    random_state=42)

x_train = np.expand_dims(x_train.values, 1).astype(np.float32)
y_train = np.expand_dims(y_train.values, 1).astype(np.float32)
x_validation = np.expand_dims(x_test.values, 1).astype(np.float32)
y_validation = np.expand_dims(y_test.values, 1).astype(np.float32)

qnn_model = VQNNModel()
log_dir = "logs\\model\\workspace\\"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
qnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     beta_1=0.9,
                                                     beta_2=0.999,
                                                     epsilon=1e-07),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=["mean_squared_error"])

qnn_model.run_eagerly = True
qnn_model.fit(x_train, y_train, epochs=10, batch_size=1, callbacks=[tensorboard_callback])
# %%
num_of_features = fdf_x_scaled.shape[1]
qnn_predictions = []
for entry in fdf_x_scaled.iterrows():
    fdf_x_predict_tensor = tf.reshape(tf.convert_to_tensor(entry[1]), [1, num_of_features])
    qnn_forecast = qnn_model.predict(fdf_x_predict_tensor)
    qnn_predictions.append(qnn_forecast[-1, -1])

signal = [0 if x <= 0.04 else 1 for x in qnn_predictions]

def calculate_pnl(signal, returns):
    pnl = [1]
    c_pnl=1
    for i, s in enumerate(signal):
        c_pnl *= (1 + s * returns[i])
        pnl.append(c_pnl)
    return pnl

signal_prx = []
for i in range(0, fdf_y_len):
    s = np.NaN
    if signal[i]:
        s = fdf_prx[i]
    signal_prx.append(s)

pnl = calculate_pnl(signal, fdf_y.to_list())

fig, ax = plt.subplots()

plt.grid(axis='both', color='0.95')
plt.xticks(rotation=45)
plt.xticks(range(0, int(fdf_y_len), 10))
ax.plot(range(len(pnl)), pnl, label='P&L flow', color='green')
ax.set_xlabel('Date')
ax.set_ylabel('P&L')
ax.set_title("Profit & Loss, price and signal plot")
ax.legend(loc='upper left')
ax_prx = ax.twinx()
ax_prx.plot(fdf_prx, label='IBM.N price', color='blue')
ax_prx.set_ylabel('Close Price')
ax_prx.plot(signal_prx, label='Signal', color='green', marker='$B$', linestyle='None')
ax_prx.legend()
fig.show()

