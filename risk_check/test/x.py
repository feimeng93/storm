import sympy as sp
import torch

# Define SymPy variables
x, y = sp.symbols('x y')

# Define a symbolic polynomial
p = 2 * x**4 + 2 * x**3 * y - x**2 * y**2 + 5 * y**4

# Convert to a lambda function (NumPy-compatible)
p_func = sp.lambdify((x, y), p, 'numpy')

# Convert to a PyTorch function
def p_torch(x_torch, y_torch):
    return torch.tensor(p_func(x_torch.numpy(), y_torch.numpy()), dtype=torch.float32)

# Example: Evaluate on PyTorch tensors
x_torch = torch.tensor([1.0, 2.0])
y_torch = torch.tensor([3.0, 4.0])
result = p_torch(x_torch, y_torch)

print("PyTorch Evaluation:", result)
