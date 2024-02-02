import csv
import glob
import pandas as ps
import numpy as ny
from ypstruct import struct
from numpy import random

# Action Categories
# 1: increasing \ automation on
# 2: increasing \ automation off
# 3: decreasing \ automation on
# 4: decreasing \ automation off
# 5: constant high \ automation on
# 6: constant high \ automation off
# 7: constant low \ automation on
# 8: constant high \ automation off


# Constants and thresholds
na = 8  # Number of actions
ns = 2  # number of states
nt = 20  # number of trials
deltaP_thresh = 168  # threshold for "constant" points
highP = 810  # threshold for high Score
highSC = 56  # threshold for high SC


# Reading all data files
RawData = []
Data_frame = []
Data = []

path = r'.\csvDataFiles'
filenames = glob.glob(path + "/*.csv")
for filename in filenames:
    RawData.append(ps.read_csv(filename))


# Concatenate all data
Data_frame = ps.concat(RawData, ignore_index=True)
Data = Data_frame.to_numpy()  # Convert pandas data frame to numpy array

np = int(len(Data)/nt)  # Number of participants


# Initializing arrays
ActionData = ny.zeros((nt-1, np))
AutoData = ny.zeros((nt, np))
PData = ny.zeros((nt, np))
SCdata = ny.zeros((nt-1, np))


# splitting data into trials
for i in range(1, np+1):

    # Array indexing
    start_idx = (nt*(i-1))
    end_idx = ((nt*i)-1)

    # Self-Confidence
    SCdata[0:nt-1, i-1] = Data[start_idx+1:end_idx+1, 7]  # split by participant
    SCdata[0:nt-1, i-1] = (SCdata[0:nt-1, i-1] >= highSC) + 1  # split high/low

    # Actions
    PData[0:nt, i-1] = Data[start_idx:end_idx+1, 1]
    AutoData[0:nt, i-1] = Data[start_idx:end_idx+1, 8]

    # Sorting Actions
    for j in range(1, nt):

        if int(AutoData[j, i-1]) == 1:   # Automation off
            CurrentAction = 2

            if abs(PData[j, i-1] - PData[j-1, i-1]) <= deltaP_thresh:  # Check if constant
                CurrentAction = CurrentAction + 4                      # Move to constant/off - 6
                ActionData[j - 1, i - 1] = CurrentAction

                if PData[j, i-1] < highP:            # Constant low/off - 8
                    CurrentAction = CurrentAction + 2
                    ActionData[j - 1, i - 1] = CurrentAction

                else:
                    CurrentAction = CurrentAction  # Constant high/off - 6
                    ActionData[j - 1, i - 1] = CurrentAction

            elif PData[j, i-1] < PData[j-1, i-1]:   # Decreasing/off - 4
                CurrentAction = CurrentAction + 2
                ActionData[j - 1, i - 1] = CurrentAction

            else:                       # Increasing/off - 2
                CurrentAction = CurrentAction
                ActionData[j - 1, i - 1] = CurrentAction
        else:
            CurrentAction = 1   # Automation on

            if abs(PData[j, i - 1] - PData[j - 1, i - 1]) <= deltaP_thresh:  # Check if constant
                CurrentAction = CurrentAction + 4                            # Move to constant/on - 5

                if PData[j, i - 1] < highP:  # Constant low/on - 7
                    CurrentAction = CurrentAction + 2

                else:
                    CurrentAction = CurrentAction  # Constant high/on - 5

            elif PData[j, i - 1] < PData[j - 1, i - 1]:  # Decreasing/on - 3
                CurrentAction = CurrentAction + 2

            else:  # Increasing/on - 1
                CurrentAction = CurrentAction

            ActionData[j-1, i-1] = CurrentAction  # Update Action array

# Participant data split by individual participant
# U: input vectors - action
U = ny.hsplit(ActionData, np)
# Y: output vectors - self-confidence
Y = ny.hsplit(SCdata, np)

# ny.set_printoptions(threshold=ny.inf)  #print entire matrix/vectors










