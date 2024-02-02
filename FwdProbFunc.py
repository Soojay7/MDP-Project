import numpy as ny
from numpy import random
import math
from vector2modelfunc import *


# Function to compute forward probability used to calculate a Fitness value for the genetic algorithm
def forwardprob(pop_vector, U, Y, na, nt, ns, np):
    ga_solution = list(pop_vector)
    loglikelihood = 0
    model = vector2model(ga_solution, na, ns)
    Pi = model[0]
    A = model[1]

    # create list of participant arrays storing alpha and P - probability
    alpha = [] * np
    P = [0] * np
    for i in range(0, np):
        alpha.append(ny.zeros((nt, ns)))

    for c in range(0, np):  # for each action set

        u = ny.insert(U[c], 0, 0)     # set u as current action set
        y = ny.insert(Y[c], 0, 0)     # set y as current SC set
        alpha[c][0] = Pi              # first state probability
        for t in range(1, nt):
            alpha[c][t] = alpha[c][t-1] * sum(A[int(y[t]-1), :, int(u[t]-1)])
        P[c] = ny.sum(alpha[c][:])
        loglikelihood = loglikelihood + math.log(P[c])

    return loglikelihood
