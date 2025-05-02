import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
from helper import create_df
import h2o
from h2o.automl import H2OAutoML
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#data = create_df()

data = pd.read_csv("data/processed/processed_TSLA.csv")
data['Date'] = pd.to_datetime(data['Date'])

columns = ['sma_20', 'ema_20', 'macd', 'rsi', 'stoch', 'bollinger_high',
       'bollinger_low', 'atr', 'obv']

sentiment = pd.read_csv("data/tsla_sentiment_aligned.csv")

# Function to align dates
def align_dates(sentiment_df, stock_df):
    aligned_sentiment = sentiment_df
    
    for date in sentiment_df['date']:
        current_date = date
        # Keep incrementing the date until we find a match in stock dates
        while current_date not in stock_df['Date']:
            current_date += pd.Timedelta(days=1)
            # If we go too far into the future, break
            if current_date > stock_df['Date'].max():
                break
        
        if current_date in stock_df['Date']:
            # Get the sentiment data for the original date
            sentiment_data = sentiment_df.loc[date]
            # Add it to the aligned dataframe with the matched date
            aligned_sentiment = pd.concat([aligned_sentiment, 
                                         pd.DataFrame(sentiment_data).T.set_index(pd.DatetimeIndex([current_date]))])
    
    return aligned_sentiment



# Merge sentiment data with stock data
data = data.join(sentiment, how='left')



data['sentiment_imputed'] = data['net_sentiment'].isna().astype(int)

# Forward fill any NaN values in sentiment columns
data['net_sentiment'] = data['net_sentiment'].fillna(0)

train_size = int(len(data) * 0.8)

train = data.iloc[:train_size]
test = data.iloc[train_size:]
final = test

date_train = train['Date']
y_train = train['Close']
X_train = train.drop(columns=['Close', 'Date'])

date_test = test['Date']
y_test = test['Close']
X_test = test.drop(columns=['Close', 'Date'])

#----------------------------------------------------------
'''                XGBOOST                  '''

# model = xgb.XGBRegressor(
#     objective='reg:squarederror',  # For regression tasks
#     n_estimators=500,  # Number of boosting rounds
#     learning_rate=0.05,  # Step size shrinkage
#     max_depth=6,  # Depth of each tree
#     subsample=0.8,  # Row sampling
#     colsample_bytree=0.8,  # Feature sampling
#     random_state=42
# )
# model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)


# y_pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

# print(f"R²: {r2:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"RMSE: {rmse:.2f}")

# plt.figure(figsize=(12, 6))
# plt.plot(date_test, y_test, label='Actual Price')
# plt.plot(date_test, y_pred, label='Predicted Price', linestyle='--')
# plt.title("XGBoost Prediction without technicals/OHLCV")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

#-----------------------------------------------------------------------
'''                   H2O AutoML                   '''

# h2o.init()

# data = h2o.H2OFrame(train)

# # Define the target and features
# target = "Close"
# features = [col for col in data.columns if col != target]

# # Set the train and test sets (80% train, 20% test split)
# train, test = data.split_frame(ratios=[.8], seed=42)

# # Initialize and run H2O AutoML
# aml = H2OAutoML(max_models=20, seed=42, max_runtime_secs=1000)  # you can adjust these settings
# aml.train(y=target, training_frame=train, validation_frame=test)

# leaderboard = aml.leaderboard
# print(leaderboard)

# model = aml.leader

# final = h2o.H2OFrame(final)

# performance = model.model_performance(final)
# print(performance)

# Make predictions on new data (test set or unseen data)

# predictions = model.predict(final)
# # If you want to get predictions as a pandas DataFrame
# predictions_df = predictions.as_data_frame()
# test_df = final.as_data_frame()
# print(f"RMSE: {performance.rmse()}")

# model_path = h2o.save_model(model=model, path="models/", force=True)
# print(f"Model saved to: {model_path}")

# date_test = test_df['Date']
# test_df = test_df['Close']

# print(predictions_df.shape)
# print(test_df.shape)
# print(date_test.shape)

# plt.figure(figsize=(12, 6))
# plt.plot(date_test, test_df, label='Actual Price')
# plt.plot(date_test, predictions_df, label='Predicted Price', linestyle='--')
# plt.title("KO Stock Price Prediction")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

#--------------------------------------------------------------------
'''             Preloaded H2O model              '''

# model = h2o.load_model("GBM_grid_1_AutoML_1_20250415_205008_model_1")

#--------------------------------------------------------------------
'''                     ARIMA Model                     '''

# # Ensure we're only using the Close price for ARIMA
# train_series = train['Close'].astype(float)
# test_series = test['Close'].astype(float)

# # Plot ACF and PACF to help determine ARIMA parameters
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
# plot_acf(train_series, ax=ax1, lags=20)
# plot_pacf(train_series, ax=ax2, lags=20)
# plt.tight_layout()
# plt.show()

# # Create and fit the ARIMA model with optimized parameters
# model = ARIMA(train_series, order=(3, 1, 2))
# model_fit = model.fit()

# # Print model summary to check diagnostics
# print(model_fit.summary())

# # Make predictions
# forecast = model_fit.forecast(steps=len(test_series))

# plt.figure(figsize=(12, 5))
# plt.plot(date_test, test_series, label='Actual Close Price')
# plt.plot(date_test, forecast, label='Forecasted Close Price', linestyle='--')
# plt.title('ARIMA Forecast vs Actual (KO Close Price)')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Calculate and print metrics
# rmse = np.sqrt(mean_squared_error(test_series, forecast))
# mae = mean_absolute_error(test_series, forecast)
# print(f'RMSE: {rmse:.2f}')
# print(f'MAE: {mae:.2f}')

#--------------------------------------------------------------
'''                 LSTM                  '''

# # Prepare the data
# close_prices = data['Close'].values.reshape(-1, 1)

# # Scale the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(close_prices)

# # Function to create sequences
# def create_sequences(data, window_size):
#     X, y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i:(i + window_size)])
#         y.append(data[i + window_size])
#     return np.array(X), np.array(y)

# # Create sequences with window size of 7
# window_size = 20
# X, y = create_sequences(scaled_data, window_size)

# print("\nData shapes after sequence creation:")
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("\nFirst few sequences:")
# print("X[0]:", X[0].flatten())
# print("y[0]:", y[0])
# print("X[1]:", X[1].flatten())
# print("y[1]:", y[1])

# # Split into train and test sets
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# print("\nData shapes after splitting:")
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

# # Verify no overlap between train and test
# print("\nChecking for data overlap:")
# print("Last X_train sequence:", X_train[-1].flatten())
# print("First X_test sequence:", X_test[0].flatten())
# print("Last y_train value:", y_train[-1])
# print("First y_test value:", y_test[0])

# # Reshape data for LSTM [samples, time steps, features]
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # Build LSTM model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
#     Dropout(0.2),
#     LSTM(50, return_sequences=False),
#     Dropout(0.2),
#     Dense(1)
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=32,
#     validation_split=0.1,
#     verbose=1
# )

# # Make predictions
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Inverse transform predictions
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# y_train_actual = scaler.inverse_transform(y_train)
# y_test_actual = scaler.inverse_transform(y_test)

# # Calculate metrics
# train_rmse = root_mean_squared_error(y_train_actual, train_predict)
# test_rmse = root_mean_squared_error(y_test_actual, test_predict)
# train_mae = mean_absolute_error(y_train_actual, train_predict)
# test_mae = mean_absolute_error(y_test_actual, test_predict)

# print("\nModel Performance:")
# print(f'Train RMSE: {train_rmse:.2f}')
# print(f'Test RMSE: {test_rmse:.2f}')
# print(f'Train MAE: {train_mae:.2f}')
# print(f'Test MAE: {test_mae:.2f}')

# # Plot the results
# plt.figure(figsize=(12, 6))

# # Plot training predictions
# train_plot = np.empty_like(close_prices)
# train_plot[:, :] = np.nan
# train_plot[window_size:window_size + len(train_predict)] = train_predict

# # Plot test predictions
# test_plot = np.empty_like(close_prices)
# test_plot[:, :] = np.nan
# test_plot[window_size + len(train_predict):window_size + len(train_predict) + len(test_predict)] = test_predict

# plt.plot(close_prices, label='Actual Price')
# plt.plot(train_plot, label='Training Predictions')
# plt.plot(test_plot, label='Test Predictions')
# plt.title('LSTM Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Plot training history
# plt.figure(figsize=(12, 4))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss During Training')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#------------------------------------------------------------------
'''               LSTM Binary Classification        '''

# Prepare the data
close_prices = data['Close'].values.reshape(-1, 1)

data = data[[ 'Open', 'High', 'Low', 'Close', 'Volume', 'net_sentiment', 'sentiment_imputed']]

data.to_csv('data/processed/final_tsla_sentiment.csv')

# Calculate price changes and create binary labels (1 for up, 0 for down)
price_changes = np.diff(data['Close'].values)

binary_labels = (price_changes > 0).astype(int)

# Create sequences of past 7 days' data for all features
window_size = 15
X, y = [], []
for i in range(window_size, len(data)-1):
    # Get all features for the past 7 days
    X.append(data.iloc[i-window_size:i].values)
    y.append(binary_labels[i-window_size])

X = np.array(X)
y = np.array(y)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = np.array([scaler.fit_transform(x) for x in X_train])
X_test_scaled = np.array([scaler.transform(x) for x in X_test])

# Reshape data for LSTM [samples, time steps, features]
n_features = X_train_scaled.shape[2]  # Number of features
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], n_features))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], n_features))

# Build LSTM model for binary classification with more features
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
train_predict_proba = model.predict(X_train_scaled)
test_predict_proba = model.predict(X_test_scaled)

# Convert probabilities to binary predictions
train_predict = (train_predict_proba > 0.5).astype(int)
test_predict = (test_predict_proba > 0.5).astype(int)

train_accuracy = accuracy_score(y_train, train_predict)
test_accuracy = accuracy_score(y_test, test_predict)
train_precision = precision_score(y_train, train_predict)
test_precision = precision_score(y_test, test_predict)
train_recall = recall_score(y_train, train_predict)
test_recall = recall_score(y_test, test_predict)
train_f1 = f1_score(y_train, train_predict)
test_f1 = f1_score(y_test, test_predict)

print("\nModel Performance:")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Train Precision: {train_precision:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Train Recall: {train_recall:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Train F1 Score: {train_f1:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Plot confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Train confusion matrix
cm_train = confusion_matrix(y_train, train_predict)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Train Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Test confusion matrix
cm_test = confusion_matrix(y_test, test_predict)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Test Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()




