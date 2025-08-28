import math

class Value:
    """ stores a single scalar value and its gradient """
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)  # previous values that contributed to this value
        self._op = _op
        self.label = label
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other, label=other)
        out = Value(
            data=self.data+other.data,
            _children=(self, other),
            _op="+",
            label=f"{self.label}+{other.label}"
        )
        def _backward():
            self.grad += 1.0 * out.grad
            out.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other, label=other)
        out = Value(
            data=self.data*other.data,
            _children=(self, other),
            _op="*",
            label=f"{self.label}*{other.label}"
        )
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # if not isinstance(other, (int, float)):
        #     raise ValueError(f"only supporting int/float powers for now, got {type(other)}")
        
        out = Value(
            data=self.data ** other,
            _children=(self,),
            _op="**",
            label=f"{self.label}**{other}"
        )
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(
            data=math.exp(self.data),
            _children=(self,),
            _op="exp",
            label=f"exp({self.label})"
        )
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        e = math.exp(2*x)
        t = (e-1)/(e+1)
        out = Value(
            data=t,
            _children=(self,),
            _op="tanh",
            label=f"tanh({self.label})"
        )
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        out = Value(
            data=max(0, x),
            _children=(self,),
            _op="ReLU",
            label=f"relu({self.label})"
        )
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        """ sort the nodes (childre) in the graph in topological order """
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
              visited.add(v)
              for child in v._prev:
                build_topo(child)
            topo.append(v)
        
        build_topo(self)
        
        # go in reverse order and apply chain rule
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rmul__(self, other): # other * self
        return self * other
    
# # 2 dimensional neuron looks like below
# # inputs x1, x2
# x1 = Value(2.0, label="x1")
# x2 = Value(0.0, label="x2")

# # weights w1, w1
# w1 = Value(-3.0, label="w1")
# w2 = Value(1.0, label="w2")

# # bias of neuron
# b = Value(6.8813735870195432, label="b")

# # n = x1*w1 + x2*w2 + b
# x1w1 = x1 * w1; x1w1.label = "x1*w1"
# x2w2 = x2 * w2; x2w2.label = "x2*w2"
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
# n = x1w1x2w2 + b; n.label = "n"

# # o = tanh(n)
# o = n.tanh(); o.label = "o"
# print(f"o: {o}")
        