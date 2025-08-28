import random
from micrograd.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
        def parameters(self):
            return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
        
    def __call__(self, x):
        # element-wise multiplication of weights and inputs -> take a sum -> add bias
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # out = act.tanh() if self.nonlin else act
        out = act.relu() if self.nonlin else act
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        # return f"{'tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):
    def __init__(self, nin, nouts):
        dim = [nin] + nouts
        self.layers = [
            Layer(dim[i], dim[i+1], nonlin=i!=len(nouts)-1) 
            for i in range(len(nouts))
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    # def zero_grad(self):
    #     for p in self.parameters():
    #         p.grad = 0.0
            

# x = [2.0, 3.0, -1.0]
# mlp = MLP(nin=3, nouts=[4, 4, 1])
# print(f"mlp(x): {mlp(x)}")