import numpy as ny
import random
import pandas as ps
import math
import csv
import CleanData as cd
from vector2modelfunc import *
import FwdProbFunc as fwd
import GAFunc as gaf
import pygad

# Dataset
ActionData = cd.U   # Action Data
SCData = cd.Y       # Self Confidence data
np = cd.np   # number of participants
na = cd.na   # number of actions
ns = cd.ns   # number of states
nt = cd.nt   # number of trials

# k-fold validation
Loglike_values = []*10
numIteration = 10
for i in range(0, numIteration):
    Kfold = 5
    NumGroup = np/Kfold
    RandGroup = ny.random.permutation(40)

    GroupsAction = []*5
    GroupsSC = []*5
    GroupIdx = []*5
    foldData = []*5
    for i in range(0,5):
        GroupsSC.append(ny.zeros((19, 8)))
        GroupsAction.append(ny.zeros((19,8)))
        foldData.append(ny.zeros((19,8)))
        GroupIdx.append(ny.zeros((8)))

    idx1 = -8
    idx2 = 0
    for i in range(0, int(Kfold)):
        #Split into groups
        idx1 = idx1 + 8
        idx2 = idx2 + 8
        GroupIdx[i] = RandGroup[idx1:idx2]

    for i in range(0, 5):
        for j in range(0, 8):
            GroupsAction[i][:,j] = ActionData[int(GroupIdx[i][j])].transpose()
            GroupsSC[i][:,j] = SCData[int(GroupIdx[i][j])].transpose()

    # test_ActionGroup = ny.zeros((19,8))
    # test_SCGroup = ny.zeros((8))

    Randtest = ny.random.randint(0,5)

    test_ActionGroup = GroupsAction[int(Randtest)]
    test_SCGroup = GroupsSC[int(Randtest)]

    # Deletes test group from all groups to form training group
    train_SCGroup = GroupsSC
    train_ActionGroup = GroupsAction
    train_SCGroup.pop(int(Randtest))
    train_ActionGroup.pop(int(Randtest))

    # Concatenated groups
    train_ActionGroupConcat = ny.concatenate(train_ActionGroup[0:4], axis=1)
    train_SCGroupConcat = ny.concatenate(train_SCGroup[0:4], axis=1)

    U_train = ny.hsplit(train_ActionGroupConcat, 32)
    Y_train = ny.hsplit(train_SCGroupConcat, 32)

    U_test = ny.hsplit(test_ActionGroup, 8)
    Y_test = ny.hsplit(test_SCGroup, 8)

    # Train Model and store best solution (Pi, A)
    Trained_model = gaf.ga_func(U_train, Y_train, na, nt, ns, 32)

    # Test Model - store loglike values for each iteration
    Test_group = fwd.forwardprob(Trained_model, U_test, Y_test, na, nt, ns, 8)
    Loglike_values.append(Test_group)


# Outputs the Average Loglikelihood
Av_Loglike = sum(Loglike_values)/len(Loglike_values)

# Calculating Root Mean Squared, Standard Deviation and Error
MS = ((Loglike_values[0] - Av_Loglike)**2+(Loglike_values[1] - Av_Loglike)**2+(Loglike_values[2] - Av_Loglike)**2+(Loglike_values[3] - Av_Loglike)**2+(Loglike_values[4] - Av_Loglike)**2+(Loglike_values[5] - Av_Loglike)**2+(Loglike_values[6] - Av_Loglike)**2+(Loglike_values[7] - Av_Loglike)**2+(Loglike_values[8] - Av_Loglike)**2+(Loglike_values[9] - Av_Loglike)**2)
RMS = (MS)**0.5
SD = (RMS/len(Loglike_values))**0.5
Error = SD/(len(Loglike_values))**0.5

print('Average Loglike of 10 iterations of 5fold cross val:', Av_Loglike, '±', Error)





