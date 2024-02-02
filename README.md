# MDP-Project

## Context

In this project, data was gathered from 40 participants. The participants were required to play a game where they tried to land a quadrotor on a landing pad successfully. The game adapts to the participants' performance and automatically toggles automation assistance on/off. Each participant had 20 trials after which they reported their self-confidence level in landing the quadrotor. (Data for each participant in separate CSV files)

## Project Goal

Create a probabilistic dynamic model to estimate and predict the human self-confidence cognitive state which can then be later used in different applications to accelerate the learning process for different tasks (e.g. in our case to land a quadrotor)

## Methodology

To build the model a Markov Decision Process (MDP) is used. However, instead of using a specific reward function, a genetic algorithm is used to obtain an optimized model.

The MDP model consists of:
Action: change in performance between trials + if automation assistance was on/off for the trial
States: if the participant was of low/high self-confidence for the trial
Transition probability: calculated using the action of the participant + the state of the participant in the previous trial

## Code Files

CleanData.py: Cleans and categorizes the data to input into the model to be optimized 

MDPmodel.py: Calculates the transition probabilities and loglikelihood for our model. The loglikelihood is then used as the fitness parameter for the genetic algorithm optimizing the model

GAtoCSV.py (Main): This is the "main" file to run. It runs the MDPmodel.py but outputs the results of the optimized model to a CSV.

5foldCrossVal.py: Validates the model by doing 10 iterations of a 5-fold cross-validation.

DataROC.py: Assesses the performance of the model by plotting a Receiver Operation Curve and calculating the area under it.

FwdProbFunc.py, GAFunc, vector2modelfunc: Function files.



