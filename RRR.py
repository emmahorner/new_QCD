import numpy as np

g = np.load("blah_data.npz")

print("Opening data file in RRR, index={}".format(g['i']))

A = g['A']
B = g['B']
C = g['C']

def print_data():
    print(np.mean(A), np.std(B), C)