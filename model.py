import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import xgboost as xgb

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Spark session
def init_spark_session():
    spark = SparkSession.builder \
        .appName("Wait Time Prediction") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    return spark

# Load data and convert to Pandas DataFrame
def load_data(spark, file_path, sample_fraction=None):
    df_spark = spark.read.parquet(file_path)
    
    if sample_fraction is not None and 0 < sample_fraction < 1:
        df_spark = df_spark.sample(withReplacement=False, fraction=sample_fraction, seed=42)
        print(f"Performed without replacement sampling, sampling ratio: {sample_fraction}")
    
    features = ["weather_code", "temp_num", "wind_num", "humidity_num", 
                "barometer_num", "visibility_num", "PULocationID", 
                "request_hour", "is_weekend"]
    target = "wait_minutes"
    
    df_pandas = df_spark.select(features + [target]).toPandas()
    
    df_pandas = df_pandas[df_pandas[target] >= 0]
    
    print(f"Data loading completed, shape: {df_pandas.shape}")
    print(f"Target variable statistics:\n{df_pandas[target].describe()}")
    
    return df_pandas, features, target


# Custom dataset for neural network
class WaitTimeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural network model with dropout layers
class WaitTimePredictor(nn.Module):
    def __init__(self, input_size):
        super(WaitTimePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Train neural network model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, train_losses, val_losses

# Evaluate neural network model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Train and evaluate XGBoost model
def train_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, features):
    print("\nStarting XGBoost model training...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 1
    }
    
    watchlist = [(dtrain, "train"), (dval, "val")]
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    y_pred = xgb_model.predict(dtest)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nXGBoost test set evaluation results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    
    return y_pred, y_test, xgb_model

# Plot training loss curves
def plot_loss_curves(train_losses, val_losses, model_name="Neural Network", save_path=None):
    if save_path is None:
        save_path = f"{model_name} loss curve.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"{model_name} loss curve saved to: {save_path}")




# Plot error distribution histogram
def plot_error_distribution(preds, labels, model_name="Model", save_path=None):
    if save_path is None:
        save_path = f"{model_name} error distribution histogram.png"
    
    errors = preds - labels
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.title(f'{model_name} Distribution of Prediction Errors')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"{model_name} error distribution plot saved to: {save_path}")

# Compare performance of two models
def compare_models(nn_preds, xgb_preds, true_labels):
    nn_mse = mean_squared_error(true_labels, nn_preds)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(true_labels, nn_preds)
    nn_r2 = r2_score(true_labels, nn_preds)
    
    xgb_mse = mean_squared_error(true_labels, xgb_preds)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_mae = mean_absolute_error(true_labels, xgb_preds)
    xgb_r2 = r2_score(true_labels, xgb_preds)
    
    comparison = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R²"],
        "Neural Network": [f"{nn_mse:.4f}", f"{nn_rmse:.4f}", f"{nn_mae:.4f}", f"{nn_r2:.4f}"],
        "XGBoost": [f"{xgb_mse:.4f}", f"{xgb_rmse:.4f}", f"{xgb_mae:.4f}", f"{xgb_r2:.4f}"]
    })
    
    print("\nModel performance comparison:")
    print(comparison)
    
    plt.figure(figsize=(10, 6))
    plt.table(
        cellText=comparison.values, 
        colLabels=comparison.columns, 
        cellLoc='center', 
        loc='center', 
        edges='closed'
    )
    plt.axis('off')
    plt.title('Model Performance Comparison', fontsize=14)
    plt.savefig("model performance comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Model performance comparison table saved to: model performance comparison.png")

def main():
    spark = init_spark_session()
    df, features, target = load_data(spark, "G:/processed_data.parquet", 0.1)
    
    X = df[features].values
    y = df[target].values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate neural network model
    print("\nStarting neural network model training...")
    train_dataset = WaitTimeDataset(X_train_scaled, y_train)
    val_dataset = WaitTimeDataset(X_val_scaled, y_val)
    test_dataset = WaitTimeDataset(X_test_scaled, y_test)
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_size = len(features)
    model = WaitTimePredictor(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=50
    )
    
    torch.save(trained_model.state_dict(), "wait_time_predictor.pth")
    print("Neural network model saved as: wait_time_predictor.pth")
    
    nn_preds, nn_labels = evaluate_model(trained_model, test_loader)
    
    nn_mse = mean_squared_error(nn_labels, nn_preds)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(nn_labels, nn_preds)
    nn_r2 = r2_score(nn_labels, nn_preds)
    
    print(f"\nNeural network test set evaluation results:")
    print(f"Mean Squared Error (MSE): {nn_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {nn_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {nn_mae:.4f}")
    print(f"R² Score: {nn_r2:.4f}")
    
    plot_loss_curves(train_losses, val_losses, "MLP")
    plot_error_distribution(nn_preds.flatten(), nn_labels.flatten(), "MLP")
    
    # Train and evaluate XGBoost model
    xgb_preds, xgb_labels, xgb_model = train_evaluate_xgboost(
        X_train_scaled, y_train, 
        X_val_scaled, y_val, 
        X_test_scaled, y_test,
        features
    )
    
    xgb_model.save_model("xgb_wait_time_predictor.model")
    print("XGBoost model saved as: xgb_wait_time_predictor.model")
    
    plot_error_distribution(xgb_preds, xgb_labels, "XGBoost")
    
    compare_models(nn_preds.flatten(), xgb_preds, y_test)
    
    spark.stop()

if __name__ == "__main__":
    main()