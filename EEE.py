import RRR
import importlib
import numpy as np

importlib.reload(RRR)

print("import EEE")

def sp():
    print("Called EEE.sp")
    return np.mean(RRR.A), np.round(RRR.C**2)

def print_A():
    print(RRR.A)
    
def print_AC():
    print(RRR.A * RRR.C)
    
def print_index():
    print(RRR.C**2)