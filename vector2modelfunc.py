import numpy as ny
import pandas as ps
import math

# Function to convert solution list to Initial probability array (Pi) and Action array (A)
def vector2model(ga_solution, na, ns):

    Pi = ny.zeros((1,2))
    A = ny.zeros((ns,ns,na))

    Pi[0,0] = ga_solution[0]
    Pi[0,1] = 1-ga_solution[0]

    idx = 0
    for k in range(0, na):
        idx = idx + 1
        A[0, 0, k] = ga_solution[idx]
        #A[0, 1, k] = 1 - ga_solution[idx]
        A[1, 0, k] = 1 - ga_solution[idx]

        idx = idx + 1
        #A[0, 1, k] = ga_solution[idx]
        A[1, 0, k] = ga_solution[idx]
        A[1, 1, k] = 1 - ga_solution[idx]
    return Pi, A