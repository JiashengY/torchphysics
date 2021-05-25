import pytest
import torch
import numpy as np
from neural_diff_eq.utils.differentialoperators import laplacian

# Test laplace-operator
def function(a):
    out = 0
    for i in range(len(a)):
        out += a[i]**2
    return out

def test_laplacian_for_one_input():
    a = torch.tensor([[1.0,1.0]], requires_grad=True)
    output = function(a[0])
    l = laplacian(output, a)
    assert l.shape[0] == 1
    assert l.shape[1] == 1
    assert l.detach().numpy()[0] == 4

def test_laplacian_for_many_inputs():
    a = torch.tensor([[1.0,1.0], [2.0,3.4], [1.3,2], [0,0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4,4,4,4])

def test_laplacian_in_1D():
    a = torch.tensor([[1.0], [2.0], [1.3]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [2,2,2])

def test_laplacian_in_3D():
    a = torch.tensor([[1.0,3.4,1.0], [2.0,0,0], [1.3,9,1]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [6,6,6])

def test_laplacian_for_complexer_function_1():
    a = torch.tensor([[1.0,1.0,1.0], [2.0,1.0,0], [0,0,0], [1.0,0,4.0]], requires_grad=True)
    def function1(a):
        return a[0]**2 + a[1]**3 + 4*a[2]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function1(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[32], [8], [2], [98]])

def test_laplacian_for_complexer_function_2():
    a = torch.tensor([[1.0,1.0], [2.0,0], [0,0], [0,4.0], [2,2]], requires_grad=True)
    def function1(a):
        return a[0]**3 + torch.sin(a[1])
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function1(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[6-np.sin(1)], [12], [0], [-np.sin(4)], [12-np.sin(2)]])

def test_laplacian_for_two_inputs_one_linear():
    a = torch.tensor([[1.0,1.0], [2.0,0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + a[1]**2 + b[0]
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function1(a[i], b[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4,4])  
    l = laplacian(output, b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [0,0])  

def test_laplacian_for_two_not_linear_inputs():
    a = torch.tensor([[1.0,1.0], [2.0,0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + a[1]**2 + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function1(a[i], b[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4,4])  
    l = laplacian(output, b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[6],[3]]) 