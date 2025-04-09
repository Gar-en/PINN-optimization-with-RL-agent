from stable_baselines3 import PPO
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from scipy.interpolate import griddata
# from PINN import PINN
from Burgers_Identification import PhysicsInformedNN
from PINNEnvironment import PINNEnvironment

# # Инициализация
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pinn = PINN().to(device)

# # Подготовка данных
# x_data = torch.linspace(0, 1, 100, device=device).view(-1, 1)
# y_data = torch.sin(x_data * 2 * np.pi)
# x_physics = torch.linspace(0, 1, 50, device=device).view(-1, 1)

# # Создание среды
# env = PINNEnvironment(pinn, x_data, y_data, x_physics)

# # Создание и обучение RL агента
# model = PPO(
#     "MlpPolicy", 
#     env,
#     verbose=1,
#     learning_rate=1e-4,
#     n_steps=256,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2
# )

# # Обучение
# model.learn(total_timesteps=10000)

# # Сохранение модели
# model.save("pinn_rl_agent")

# Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nu = 0.01/np.pi

N_u = 2000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0) 

# create training set
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx,:]
u_train = u_star[idx,:]

weights = np.array([1.0, 1.0])

# training
# model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)

pinn = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, weights)

# Данные
# x_data = torch.linspace(0, 1, 100).view(-1,1).to(device)
# y_data = torch.sin(x_data).to(device)
# data = (x_data, y_data)

# Физическое уравнение (пример: -u'' = f(x))
# physics_eq = lambda x, u: -torch.sin(x)  # Просто пример

# Среда
env = PINNEnvironment()

# RL агент
model = PPO("MlpPolicy", env, verbose=1)

pinn.train(1000, env, model, weights)

# Оптимизатор для PINN
# optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)

# Обучение
# for epoch in range(100):
#     # Получаем действие от агента
#     state = env.reset() if epoch == 0 else state
#     action, _ = model.predict(state, deterministic=True)
    
#     # Шаг среды (обновляет веса в PINN)
#     state, reward, done, _ = env.step(action)
    
#     # Обновляем PINN
#     # optimizer.zero_grad()
#     # loss = pinn.compute_loss(data, pinn.weights)
#     # loss.backward()
#     # optimizer.step()
    
#     # Обучаем агента на этом опыте
#     model.learn(total_timesteps=1)
    
#     if done:
#         print(f"Training finished at epoch {epoch}")
#         break

    # if epoch % 10 == 0:
    # print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Weights: {pinn.weights.detach().numpy()}")