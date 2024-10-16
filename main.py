from graphviz import Digraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace


# ____    ____  ___       __       __    __   _______      ______   .______          __   _______   ______ .___________.
# \   \  /   / /   \     |  |     |  |  |  | |   ____|    /  __  \  |   _  \        |  | |   ____| /      ||           |
#  \   \/   / /  ^  \    |  |     |  |  |  | |  |__      |  |  |  | |  |_)  |       |  | |  |__   |  ,----'`---|  |----`
#   \      / /  /_\  \   |  |     |  |  |  | |   __|     |  |  |  | |   _  <  .--.  |  | |   __|  |  |         |  |
#    \    / /  _____  \  |  `----.|  `--'  | |  |____    |  `--'  | |  |_)  | |  `--'  | |  |____ |  `----.    |  |
#     \__/ /__/     \__\ |_______| \______/  |_______|    \______/  |______/   \______/  |_______| \______|    |__|


def create_value(data, _children=(), _op=None, label=''):
    value = SimpleNamespace(
        data=data,
        _prev=list(_children),
        _op=_op,
        label=label

    )

    def print():
        return f"Value(data:{value.data})"
    value.print = print

    def add(other):
        out = create_value(value.data + other.data,
                           _children=(value, other), _op='+')
        return out
    value.add = add

    def mul(other):
        out = create_value(value.data * other.data,
                           _children=(value, other), _op='*')
        return out
    value.mul = mul

    return value

#  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,
# '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'ii


a = create_value(2.0, label='a')
b = create_value(-3.0, label='b')
c = create_value(10.0, label='c')
d = a.mul(b)
e = d.add(c)
d.label = 'd'
e.label = 'e'

#  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,  ,d88b.    ,
# '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'  '    `Y88P'


# ____    ____  __       _______. __    __       ___       __
# \   \  /   / |  |     /       ||  |  |  |     /   \     |  |
#  \   \/   /  |  |    |   (----`|  |  |  |    /  ^  \    |  |
#   \      /   |  |     \   \    |  |  |  |   /  /_\  \   |  |
#    \    /    |  | .----)   |   |  `--'  |  /  _____  \  |  `----.
#     \__/     |__| |_______/     \______/  /__/     \__\ |_______|


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
        dot.node(name=uid, label="{ %s | data %.4f }" % (
            n.label, n.data), shape='record')
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
