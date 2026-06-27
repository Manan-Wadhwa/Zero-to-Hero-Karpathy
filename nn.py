#%%
import math 
import numpy as np
import matplotlib.pyplot as plt

# %%
def f(x):
    return 3*x**2 - 4*x + 5
#%%
f(3.0)
xs = np.arange(-5.0,5.0,0.25)
ys = f(xs)
plt.plot(xs,ys)

# %%
# the derivative is the slope tell the direction and rate of change
a = 3.0
b = 4.0
c = 5.0
d = a*b + c
# the slope with respect to a would be little change diveded by the change
h = 0.00001
d1 = (a + h) *b + c
derivative = (d1 - d)/h
derivative
# %%
class Value:
    def __init__(self,data, _children = (), _op = '', label = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        out = Value(self.data + other.data, (self,other), '+')
        return out
    def __mul__(self,other):
        out = Value(self.data * other.data , (self,other), '*')
        return out
a = Value(3.0)
b = Value(2.0)
c = Value(4.0)
d = a * b + c
d
# %%
d._op
#%%
d._prev
# %%