import numpy as ny
import pandas as ps
import math
import CleanData as cd
import matplotlib.pyplot as plt

np = cd.np  # number of participants
nt = cd.nt  # number of trials
na = cd.na  # number of actions
ns = cd.ns  # number of states
Y = cd.Y    # Self Confidence vector
U = cd.U    # Action Vector
b_low = ny.zeros((nt, np))   # low belief
b_high = ny.zeros((nt, np))  # high belief
b = ny.zeros((nt, np))       # Belief
steps = 100
TPR_l = []*steps             # True Positive Rate
FPR_l = []*steps             # False Positice Rate
Thresholds = ny.linspace(0,1,num=steps)  # thresholds


def vector2model(ga_solution, na, ns):

    Pi = ny.zeros((1,2))
    A = ny.zeros((ns,ns,na))

    Pi[0,0] = ga_solution[0]
    Pi[0,1] = 1-ga_solution[0]

    idx = 0
    for k in range(0, na):
        idx = idx + 1
        A[0, 0, k] = ga_solution[idx]
        A[1, 0, k] = 1 - ga_solution[idx]

        idx = idx + 1
        A[0, 1, k] = ga_solution[idx]
        A[1, 1, k] = 1 - ga_solution[idx]
    return Pi, A

# Model obtained from FinalOutput.csv after running GAtoCSV.py through completion (250 generations)
model = [0.2435,0.9995,0.9944,0.9997,0.9977,0.9976,0.9996,0.9957,0.9975,0.9994,0.9997,0.578,0.2161,0.9908,0.978286145,0.9702,0.9911]

# Breaking model into Initial Probability array (Pi) and Action array (A)
Convert_model = vector2model(model, na, ns)
Pi = Convert_model[0]
A = Convert_model [1]

# Calculating beliefs (b) based on the model
for p in range(0, np):
    u = ny.insert(U[p], 0, 0)  # set u as current action set

    b_low[0, p] = 1
    b_high[0, p] = 1

    for i in range(1, nt):

        # calculate b for both high and low SC:
        b_low[i,p] = ((b_low[i-1,p]) * A[int(Y[p][i-1])-1,0,int(u[i])-1]) / (sum([A[0,0,int(u[i])-1] * b_low[i-1,p], A[1,0,int(u[i])-1] * b_high[i-1,p]]))
        b_high[i,p] = ((b_high[i-1,p]) * A[int(Y[p][i-1])-1,1,int(u[i])-1]) / (sum([A[0,1,int(u[i])-1] * b_low[i-1,p], A[1,1,int(u[i])-1] * b_high[i-1,p]]))

        # Choosing b_low or b_high
        if int(Y[p][i-1]) == 1:
            b[i, p] = b_low[i, p]
        else:
            b[i, p] = 1 - b_high[i, p]

# Calculated belief array
b1 = ny.delete(b, 0, axis=0)


# Check for True/False positives/negative using calculated beliefs
for t in range(0, steps):
    th = Thresholds[t]
    TP = 0   # True Positive
    FP = 0   # False Positive
    TN = 0   # True Negative
    FN = 0   # False Negative
    for p in range(0, np):
        for j in range(0, nt-1):
            if b1[j, p] >= th and int(Y[p][j]) == 1:
                TP = TP + 1
            elif b1[j, p] >= th and int(Y[p][j]) != 1:
                FP = FP + 1
            elif b1[j, p] < th and int(Y[p][j]) == 1:
                FN = FN + 1
            elif b1[j, p] < th and int(Y[p][j]) != 1:
                TN = TN + 1

            print(TP, FP, FN, TN)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TPR_l.append(TPR)
    FPR_l.append(FPR)


# Print Receiver Operating Characteristic (ROC) graph
plt.plot(FPR_l, TPR_l, label="Self-Confidence, AUC = 0.7872")
plt.plot([0,1], [0,1], label="Random Guess", linestyle='dashed')
plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.show()

# Calculate and print Area Under Curve (AUC)
AUC = -(ny.trapz(TPR_l, FPR_l))
print('AUC:', AUC)