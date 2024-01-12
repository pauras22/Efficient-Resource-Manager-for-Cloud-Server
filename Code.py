# Download data from kaggle
! pip install kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
# dataset download
# https://www.kaggle.com/datasets/gauravdhamane/gwa-bitbrains
! kaggle datasets download gauravdhamane/gwa-bitbrains

!unzip gwa-bitbrains.zip

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler

# univariate lstm 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from datetime import datetime

df=pd.read_csv("/content/fastStorage/2013-8/102.csv",sep=";\t")

df.head()

df.describe()

df["date_time"]=df["Timestamp [ms]"].apply(lambda x: datetime.fromtimestamp(x))

df.head(5)

df.drop_duplicates(subset=['date_time'], keep=False,inplace=True)

df.info()

df["Disk I/O [KB/s]"]=df["Disk read throughput [KB/s]"]+df["Disk write throughput [KB/s]"]
df["Network Traffic [KB/s]"]=df["Network received throughput [KB/s]"]+df["Network transmitted throughput [KB/s]"]

df.head()

new_df=df[["date_time","CPU usage [MHZ]","Memory usage [KB]","Disk I/O [KB/s]","Network Traffic [KB/s]"]]

new_df.head()

import plotly.express as px
plt.figure(figsize=(16,9))
fig = px.line(new_df, x=new_df.date_time, y="CPU usage [MHZ]", title='CPU usage')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

import plotly.express as px
plt.figure(figsize=(16,9))
fig = px.line(new_df, x=new_df.date_time, y="Memory usage [KB]", title='Memory usage')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

import plotly.express as px
plt.figure(figsize=(16,9))
fig = px.line(new_df, x=new_df.date_time, y="Disk I/O [KB/s]", title='Disk I/O')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

import plotly.express as px
plt.figure(figsize=(16,9))
fig = px.line(new_df, x=new_df.date_time, y="Network Traffic [KB/s]", title='Network Traffic')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# set date column as index
new_df.set_index('date_time', inplace=True)

# resample to minute frequency
new_df = new_df.resample('T').mean()

# impute missing values with interpolation
new_df.interpolate(method='linear', inplace=True)

new_df.head()

scaler = MinMaxScaler()

scaled_features = pd.DataFrame(scaler.fit_transform(new_df), columns=new_df.columns, index=new_df.index)

print(scaled_features)

# split the data into training and testing sets
train_size = int(len(scaled_features) * 0.8)
train_data = scaled_features.iloc[:train_size]
test_data = scaled_features.iloc[train_size:]

seq_length = 30 # length of input sequence
def create_sequences(data):
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# define the model architecture
model = Sequential([
    LSTM(units=64, input_shape=(seq_length, 4), activation='relu', return_sequences=True),
    LSTM(units=32, activation='relu'),
    Dense(units=4)
])

# compile the model
model.compile(optimizer='adam', loss='mae')
model.summary()

# train the model
from tensorflow import keras
early_stopping =  keras.callbacks.EarlyStopping(patience=2,monitor="val_loss", restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.25,callbacks=[early_stopping])

# evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(test_loss)

# make predictions on the test data
y_pred = model.predict(X_test)
print(y_pred)

y_pred.shape

# Inverse transform the scaled data to get the actual values
y_pred1=scaler.inverse_transform(y_pred)

y_train1=scaler.inverse_transform(y_train)

y_test1=scaler.inverse_transform(y_test)

new_df.head()

new_df.shape

import plotly.graph_objects as go

# create a dataframe with actual and predicted values
df_results = pd.DataFrame({
    'timestamp': test_data.index[seq_length:],
    'actual_cpu': y_test1[:, 0],
    'predicted_cpu': y_pred1[:, 0],
    'actual_memory': y_test1[:, 1],
    'predicted_memory': y_pred1[:, 1],
    'actual_disk_io': y_test1[:, 2],
    'predicted_disk_io': y_pred1[:, 2],
    'actual_network_traffic': y_test1[:, 3],
    'predicted_network_traffic': y_pred1[:, 3]
})

# create a plotly figure for CPU
fig_cpu = go.Figure()

# add the actual cpu values to the plot
fig_cpu.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['actual_cpu'],
        mode='lines',
        name='Actual CPU'
    )
)

# add the predicted cpu values to the plot
fig_cpu.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['predicted_cpu'],
        mode='lines',
        name='Predicted CPU'
    )
)

# set the layout of the cpu plot
fig_cpu.update_layout(
    title='CPU Demand Forecasting',
    xaxis_title='Timestamp',
    yaxis_title='CPU Demand',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# show the cpu plot
fig_cpu.show()

# create a plotly figure for memory
fig_memory = go.Figure()

# add the actual memory values to the plot
fig_memory.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['actual_memory'],
        mode='lines',
        name='Actual Memory'
    )
)

# add the predicted memory values to the plot
fig_memory.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['predicted_memory'],
        mode='lines',
        name='Predicted Memory'
    )
)

# set the layout of the memory plot
fig_memory.update_layout(
    title='Memory Demand Forecasting',
    xaxis_title='Timestamp',
    yaxis_title='Memory Demand',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# show the memory plot
fig_memory.show()

# DISK

# create a plotly figure for Disk I/O
fig_disk = go.Figure()

# add the actual memory values to the plot
fig_disk.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['actual_disk_io'],
        mode='lines',
        name='Actual Disk IO'
    )
)

# add the predicted disk_io values to the plot
fig_disk.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['predicted_disk_io'],
        mode='lines',
        name='Predicted Disk IO'
    )
)

# set the layout of the disk plot
fig_disk.update_layout(
    title='Disk I/O Forecasting',
    xaxis_title='Timestamp',
    yaxis_title='disk_io Demand',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# show the disk_io plot
fig_disk.show()

# NETWORK TRAFFIC

# create a plotly figure for Network Traffic
fig_network = go.Figure()

# add the actual memory values to the plot
fig_network.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['actual_network_traffic'],
        mode='lines',
        name='Actual Network Traffic'
    )
)

# add the predicted Network traffic values to the plot
fig_network.add_trace(
    go.Scatter(
        x=df_results['timestamp'],
        y=df_results['predicted_network_traffic'],
        mode='lines',
        name='Predicted Network Traffic'
    )
)

# set the layout of the Network Traffic plot
fig_network.update_layout(
    title='Network Traffic Forecasting',
    xaxis_title='Timestamp',
    yaxis_title='Network Traffic',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# show the network plot
fig_network.show()
