import numpy as np
import gym
import gym.spaces
#from gym import spaces
import torch
import torch.nn as nn
from PINN import PINN
from stable_baselines3 import PPO

class PINNEnvironment(gym.Env):
    # def __init__(self, pinn, x_data, y_data, x_physics):
    #     super(PINNEnvironment, self).__init__()
    #     self.pinn = pinn
    #     self.x_data = x_data
    #     self.y_data = y_data
    #     self.x_physics = x_physics
    #     self.optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        
    #     # Action space: изменения для коэффициентов loss
    #     self.action_space = spaces.Box(
    #         low=np.array([0.1, 0.1, 0.01]), 
    #         high=np.array([10.0, 10.0, 1.0]),
    #         dtype=np.float32)
        
    #     # Observation space: текущие коэффициенты + последний loss
    #     self.observation_space = spaces.Box(
    #         low=-np.inf, 
    #         high=np.inf, 
    #         shape=(4,), 
    #         dtype=np.float32)
        
    # def step(self, action):
    #     # Устанавливаем новые коэффициенты
    #     with torch.no_grad():
    #         self.pinn.w_data.data = torch.tensor(action[0], dtype=torch.float32)
    #         self.pinn.w_phys.data = torch.tensor(action[1], dtype=torch.float32)
    #         self.pinn.w_reg.data = torch.tensor(action[2], dtype=torch.float32)
        
    #     # Оптимизируем PINN с новыми коэффициентами
    #     self.optimizer.zero_grad()
    #     loss = self.pinn.compute_loss(self.x_data, self.y_data, self.x_physics)
    #     loss.backward()
    #     self.optimizer.step()
        
    #     # Награда - уменьшение loss
    #     reward = -loss.item()
        
    #     # Состояние: коэффициенты + текущий loss
    #     state = np.array([
    #         action[0], 
    #         action[1], 
    #         action[2], 
    #         loss.item()
    #     ], dtype=np.float32)
        
    #     # Условие завершения
    #     done = loss.item() < 0.001
        
    #     return state, reward, done, {}
    
    # def reset(self):
    #     # Сбрасываем коэффициенты к начальным значениям
    #     with torch.no_grad():
    #         self.pinn.w_data.data = torch.tensor(1.0)
    #         self.pinn.w_phys.data = torch.tensor(1.0)
    #         self.pinn.w_reg.data = torch.tensor(0.1)
        
    #     # Вычисляем начальный loss
    #     initial_loss = self.pinn.compute_loss(
    #         self.x_data, 
    #         self.y_data, 
    #         self.x_physics)
        
    #     return np.array([1.0, 1.0, 0.1, initial_loss.item()], dtype=np.float32)
    def __init__(self):
        super(PINNEnvironment, self).__init__()
        self.previous_loss = None
        self.current_weights = np.array([1.0, 1.0])
        # self.state = np.array([self.current_weights], [0.0])
        # self.pinn = pinn_model
        # self.data = data
        # self.physics_eq = physics_equation
        # self.previous_loss = 0.0
        # self.weights = np.array([1.0, 1.0, self.previous_loss])
        #self.weights = weights

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    
    def step(self, action, model_RL, current_loss):

        if self.previous_loss is not None:
            reward = self.previous_loss - current_loss
        else:
            reward = 0.0

        self.previous_loss = current_loss

        state = np.concatenate((self.current_weights, current_loss), axis=None)

        done = current_loss < 0.001 or not np.isfinite(current_loss)

        info = {}
    
        return state, reward, done, info
        # self.current_loss = loss
        # if self.previous_loss != 0.0:
        #     reward = self.previous_loss - self.current_loss
        # else:
        #     reward = 0.0
        # self.previous_loss = self.current_loss

        # self.pinn.weights = torch.tensor(action, dtype=torch.float32)

        # state = np.concatenate([action, [self.current_loss]])
        # done = self.current_loss < 0.001 or not np.isfinite(self.current_loss)
        # info = {}
        # return state, reward, done, info
    
    def reset(self):
        self.previous_loss = None
        self.current_weights = np.array([1.0, 1.0])
        return np.concatenate([self.current_weights, [0.0]])

    # def learn_process(self, model_RL, current_loss):

    #     action, _ = model_RL.predict(state, deterministic=True)

    #     state, reward, done, _ = env.step(action, current_loss)


    #     self.current_weights = action

    #     if self.previous_loss is not None:
    #         reward = self.previous_loss - current_loss
    #     else:
    #         reward = 0.0
    #     self.previous_loss = current_loss

    #     model_RL.learn(total_timesteps=1)

    #     return action

    def get_weights(action):
        return action


        

        # if done:
        #     print(f"Training finished at epoch {epoch}")
        
        # print(f"Hi from RL-agent {epoch}")
        # return weights_new
    

    # def get_state(self):
    #     return np.concatenate([self.weights, [self.previous_loss]])
    
    # def is_training_done(self):
    #     return self.previous_loss < 0.01