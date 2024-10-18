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


# +  # ++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:++


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
        other = other if isinstance(
            other, SimpleNamespace) else create_value(other)
        out = create_value(value.data + other.data,
                           _children=(value, other), _op='+')

        def backward():
            value.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = backward
        return out
    value.add = add

    def mul(other):
        other = other if isinstance(
            other, SimpleNamespace) else create_value(other)
        out = create_value(value.data * other.data,
                           _children=(value, other), _op='*')

        def backward():
            value.grad += out.grad * other.data
            other.grad += out.grad * value.data
        out._backward = backward

        return out
    value.mul = mul

    # def rmul(other):
    #     assert isinstance(other, (int, float)
    #                       ), "only supporting int and float for now"

    # value.rmul = rmul

    def div(other):
        other = other if isinstance(
            other, SimpleNamespace) else create_value(other)
        out = create_value(value.data / other.data,
                           _children=(value, other), _op='/')

        def backward():
            value.grad += 1 / other.data * out.grad
            other.grad += -value.data / (other.data ** 2) * out.grad
        out._backward = backward

        return out

    value.div = div

    def neg():
        return value.mul(-1)
    value.neg = neg

    def sub(other):
        other = other if isinstance(
            other, SimpleNamespace) else create_value(other)

        return value.add(other.neg())
    value.sub = sub

    def pow(other):
        assert isinstance(other, (int, float)
                          ), "only supporting int and float for now"

        out = create_value(value.data**other,
                           _op=f"**{other}", _children=(value,))

        def _backward():
            value.grad += other * (value.data**(other-1)) * out.grad
        out.backward = _backward
        value.pow = pow
        return out

    value.pow = pow

    def exp():
        x = value.data
        out = create_value(math.exp(x), _op='expo', _children=(value,))

        def backward():
            value.grad += out.grad * out.data

        out._backward = backward
        return out

    value.exp = exp

    def tanh():
        n = value.data
        tanh = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = create_value(tanh, _op='tanh', _children=(value,))

        def backward():
            value.grad = 1 - (out.data**2)
        out._backward = backward

        return out
    value.tanh = tanh

    def autobackprop():
        topo = []

        visited = []

        def build_topo(value):
            if value not in visited:
                visited.append(value)
                for child in value._prev:
                    build_topo(child)
            topo.append(value)

        build_topo(value)

        value.grad = 1
        for node in reversed(topo):
            node._backward()
    value.autobackprop = autobackprop
    return value


# +  # ++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:++


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

# tanh() directly
o = n.tanh()
o.label = 'o'


# breaking up tanh()
e1 = n.rmul(2)
e = e1.exp()
numerator = e.sub(1)
denominator = e.add(1)
o = numerator.div(denominator)
o.label = 'o'
o.grad = 1
o.autobackprop()
draw_dot(o)

# +  # ++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:+++#++:++#++:++
