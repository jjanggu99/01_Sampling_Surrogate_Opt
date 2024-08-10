import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras import layers, models
import torch
import gpytorch
from torch import nn
import torchdiffeq

# GPU-Compatible Model Functions
def build_dnn_model(X_train, y_train, X_test, y_test, modeloutput_path):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train)
    X_test_scaled = input_scaler.transform(X_test)
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)

    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(y_train.shape[1]))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
    
    model.save(modeloutput_path + '/DNN_model.h5')

def build_cnn_model(X_train, y_train, X_test, y_test, modeloutput_path): 
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train)
    X_test_scaled = input_scaler.transform(X_test)
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)
    
    X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)
    
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
    
    model.save(modeloutput_path + '/CNN_model.h5')

class DeepKernel(nn.Module):
    def __init__(self, input_dim):
        super(DeepKernel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        return self.network(x)
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
def build_gp_dkl_model(X_train, y_train, X_test, y_test, modeloutput_path):
    # Assuming PyTorch and gpytorch are used
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    deep_kernel = DeepKernel(input_dim=X_train.shape[1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X_train_tensor, y_train_tensor, likelihood, deep_kernel)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    torch.save(model.state_dict(), modeloutput_path + '/GP_DKL_model.pth')

class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, input_dim)
        )

    def forward(self, t, y):
        return self.net(y)
def build_neural_ode_model(X_train, y_train, X_test, y_test, modeloutput_path):
    input_dim = X_train.shape[1]
    ode_func = ODEFunc(input_dim)
    y0 = torch.tensor(X_train, dtype=torch.float32)
    t = torch.tensor(np.linspace(0., 1., y_train.shape[0]), dtype=torch.float32)
    
    # Run the ODE solver
    pred_y = torchdiffeq.odeint(ode_func, y0, t)

    # You would typically define a loss function and optimize here
    # For demonstration, we just save the model function
    torch.save(ode_func.state_dict(), modeloutput_path + '/Neural_ODE_model.pth')

class PINNModel(nn.Module):
    def __init__(self, input_dim):
        super(PINNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)
def build_pinn_model(X_train, y_train, X_test, y_test, modeloutput_path):
    model = PINNModel(X_train.shape[1])
    # Define custom loss based on physical laws here

    # Example: We assume some dummy training process
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):
        # Simulate some loss computation
        optimizer.zero_grad()
        loss = torch.mean((model(torch.tensor(X_train, dtype=torch.float32)) - torch.tensor(y_train, dtype=torch.float32)) ** 2)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), modeloutput_path + '/PINN_model.pth')

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar
def build_vae_model(X_train, y_train, X_test, y_test, modeloutput_path):
    model = VAE(input_dim=X_train.shape[1], latent_dim=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(torch.tensor(X_train, dtype=torch.float32))
        recon_loss = nn.functional.mse_loss(recon_batch, torch.tensor(X_train, dtype=torch.float32))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), modeloutput_path + '/VAE_model.pth')

# CPU-Bound Model Functions
def build_random_forest_model(X_train, y_train, modeloutput_path):
    model = RandomForestRegressor(n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    joblib.dump(model, modeloutput_path + '/Random_Forest_model.pkl')

def build_xgboost_model(X_train, y_train, modeloutput_path):
    model = XGBRegressor(n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0]
    }
    search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    joblib.dump(model, modeloutput_path + '/XGBoost_model.pkl')

def build_lightgbm_model(X_train, y_train, modeloutput_path):
    model = LGBMRegressor(n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': np.linspace(0.01, 0.2, 10),
        'max_depth': [3, 5, 8],
        'num_leaves': [31, 63, 127, 255],
        'boosting_type': ['gbdt', 'dart', 'goss']
    }
    search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    joblib.dump(model, modeloutput_path + '/LightGBM_model.pkl')

def build_catboost_model(X_train, y_train, modeloutput_path):
    model = CatBoostRegressor(verbose=0, thread_count=-1)
    param_grid = {
        'iterations': [100, 500, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 10],
        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS']
    }
    search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    joblib.dump(model, modeloutput_path + '/CatBoost_model.pkl')

def Surrogate_model_build(model_type, input_data, output_data, modeloutput_path):

    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
    # GPU-Compatible Model Functions
    if model_type == 'DNN':
        build_dnn_model(X_train, y_train, X_test, y_test, modeloutput_path)
    elif model_type == 'CNN':
        build_cnn_model(X_train, y_train, X_test, y_test, modeloutput_path)
    elif model_type == 'GP-DKL':
        build_gp_dkl_model(X_train, y_train, X_test, y_test, modeloutput_path)
    elif model_type == 'Neural_ODE':
        build_neural_ode_model(X_train, y_train, X_test, y_test, modeloutput_path)
    elif model_type == 'PINN':
        build_pinn_model(X_train, y_train, X_test, y_test, modeloutput_path)
    elif model_type == 'VAE':
        build_vae_model(X_train, y_train, X_test, y_test, modeloutput_path)

    # CPU-Bound Model Functions
    elif model_type == 'Random_Forest':
        build_random_forest_model(X_train, y_train, modeloutput_path)
    elif model_type == 'XGBoost':
        build_xgboost_model(X_train, y_train, modeloutput_path)
    elif model_type == 'LightGBM':
        build_lightgbm_model(X_train, y_train, modeloutput_path)
    elif model_type == 'CatBoost':
        build_catboost_model(X_train, y_train, modeloutput_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Model {model_type} trained and saved at {modeloutput_path}.")