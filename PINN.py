import torch
import torch.nn as nn

class PINN(nn.Module):
    # def __init__(self):
    #     super(PINN, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(1, 20),
    #         nn.Tanh(),
    #         nn.Linear(20, 20),
    #         nn.Tanh(),
    #         nn.Linear(20, 1)
    #     )
        
    #     # Коэффициенты для компонент loss
    #     self.w_data = nn.Parameter(torch.tensor(1.0))
    #     self.w_phys = nn.Parameter(torch.tensor(1.0))
    #     self.w_reg = nn.Parameter(torch.tensor(0.1))
        
    # def forward(self, x):
    #     return self.net(x)
    
    # def compute_loss(self, x_data, y_data, x_physics):
    #     # Data loss
    #     y_pred = self.forward(x_data)
    #     data_loss = torch.mean((y_pred - y_data)**2)
        
    #     # Physics loss
    #     x_physics.requires_grad_(True)
    #     u = self.forward(x_physics)
    #     u_x = torch.autograd.grad(u, x_physics, create_graph=True)[0]
    #     u_xx = torch.autograd.grad(u_x, x_physics, create_graph=True)[0]
    #     physics_loss = torch.mean(u_xx**2)  # Пример для u_xx = 0
        
    #     # Regularization
    #     reg_loss = torch.sum(torch.stack([torch.norm(p) for p in self.parameters()]))
        
    #     return (self.w_data * data_loss + 
    #             self.w_phys * physics_loss + 
    #             self.w_reg * reg_loss)

    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.weights = torch.tensor([1.0, 1.0, 1.0], requires_grad=False)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def compute_loss(self, data, physics_equation, weights):
        # Data loss
        x_data, y_data = data
        y_pred = self.forward(x_data)
        data_loss = torch.mean((y_pred - y_data) ** 2)

        # Physics loss
        x_physics = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)
        u = self.forward(x_physics)
        u_x = torch.autograd.grad(u.sum(), x_physics, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x_physics, create_graph=True)[0]
        physics_loss = torch.mean((u_xx - physics_equation(x_physics, u)) ** 2)

        # Regularization (пример)
        regularization = torch.sum(torch.stack([torch.norm(param) for param in self.parameters()]))

        # Общий loss
        total_loss = weights[0] * data_loss + weights[1] * physics_loss + weights[2] * regularization
        return total_loss