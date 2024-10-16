from graphviz import Digraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import math


# +--------------+
# | visual nodes |
# +--------------+

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = [], []  # Use lists instead of sets

    def build(v):
        if v not in nodes:
            nodes.append(v)
            for child in v._prev:
                edges.append((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={
                  'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # Create a rectangular ('record') node for each value
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (
            n.label, n.data, n.grad), shape='record')
        if n._op:
            # Create an op node for the operation if exists
            dot.node(name=uid + n._op, label=n._op)
            # Connect this operation node to the value node
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

#  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,
# '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'


# +---------------------------------+
# | value object with functionality |
# +---------------------------------+

def create_value(data, _children=(), _op=None, label=''):
    value = SimpleNamespace(
        data=data,
        _prev=list(_children),
        _op=_op,
        label=label,
        grad=0,
        _backward=lambda: None

    )

    def print():
        return f"Value(data:{value.data})"
    value.print = print

    def add(other):
        out = create_value(value.data + other.data,
                           _children=(value, other), _op='+')

        def backward():
            value.grad = 1 * out.grad
            other.grad = 1 * out.grad
        out._backward = backward
        return out
    value.add = add

    def mul(other):
        out = create_value(value.data * other.data,
                           _children=(value, other), _op='*')

        def backward():
            value.grad = out.grad * other.data
            other.grad = out.grad * value.data
        out._backward = backward

        return out
    value.mul = mul

    def tanh():
        n = value.data
        tanh = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = create_value(tanh, _op='tanh', _children=(value,))

        def backward():
            value.grad = 1 - (out.data**2)
        out._backward = backward

        return out
    value.tanh = tanh

    return value

#  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,
# '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'ii


# +--------------+
# | Basic Practice
# +--------------+

a = create_value(2.0, label='a')
b = create_value(-3.0, label='b')
c = create_value(10.0, label='c')
d = a.mul(b)
d.label = 'd'
e = d.add(c)
e.label = 'e'
f = create_value(-2.0, label='f')
L = e.mul(f)
L.label = 'L'
draw_dot(L)

# +------------------------+
# | Basic Neuron calculation
# +------------------------+

# inputs x1,x2
x1 = create_value(2.0, label='x1')
x2 = create_value(0.0, label='x2')
# weights w1,w2
w1 = create_value(-3.0, label='w1')
w2 = create_value(1.0, label='w2')
# bias of the neuron
b = create_value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b

# calculation
x1w1 = x1.mul(w1)
x1w1.label = 'x1*w1'
x2w2 = x2.mul(w2)
x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1.add(x2w2)
x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2.add(b)
n.label = 'n'
o = n.tanh()
o.label = 'o'

# backpropagation manually
o.grad = 1
o._backward()
n._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()
draw_dot(o)


#  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,
# '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'
