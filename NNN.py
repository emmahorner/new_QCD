import RRR
import EEE
import numpy as np
import importlib

importlib.reload(RRR)
importlib.reload(EEE)

ABC = np.mean(RRR.A) * np.std(RRR.B) * RRR.C
print("NNN loaded with ABC = {}".format(ABC))

def solve():
    print("Called NNN.solve")
    return EEE.sp()

def print_ABC():
    print(ABC)