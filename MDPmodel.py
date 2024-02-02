import numpy as ny
from numpy import random
import CleanData as cd
import pygad
import pandas as ps
import math
from vector2modelfunc import *
import FwdProbFunc as fwd

lowSC = 1  # Low Self Confidence
highSC = 2  # High Self Confidence
possibleStates = [lowSC, highSC]
possibleActions = [1, 2, 3, 4, 5, 6, 7, 8]
ns = len(possibleStates)
na = len(possibleActions)
U = cd.U  # Importing action sequences from CleanData.py
Y = cd.Y  # Importing Self-confidence (SC) sequences from CleanData.py
nt = len(U[0])  # number of trials
np = len(U)  # number of participants


# Defining fitness function for Genetic Algorithm (GA)
def fitness_func(solution, solution_idx):
    fitness = fwd.forwardprob(solution, U, Y, na, nt, ns, np)
    return fitness


# Creating population of solutions
num_solutions = 100  # 500
pop_vector_int = ny.zeros(shape=(num_solutions, 17))

for solution_idx in range(num_solutions):
    r = ny.random.uniform(low=0, high=1, size=(17))
    pop_vector_int[solution_idx, :] = r

# GA parameters
num_generations = 300
num_parents_mating = 25  # 250

sol_per_pop = num_solutions
pop_size = num_solutions
num_genes = 17  # same as number of inputs

init_range_low = 0
init_range_high = 1

parent_selection_type = "sss"
keep_parents = -1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

Fitness = [] * num_generations
Generations = [] * num_generations
best_solution = [] * num_generations


# GA function for printing generation number and fitness after each generation iterated
def callback_gen(ga_instance):
    # Storing generation num, Fitness and best solution for .csv output file
    Generations.append(ga_instance.generations_completed)
    Fitness.append(ga_instance.best_solution()[1])
    best_solution.append(ga_instance.best_solution()[0])

    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       initial_population=pop_vector_int,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_gen,
                       parallel_processing=8,
                       gene_space=[ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001),
                                   ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001),
                                   ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001),
                                   ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001),
                                   ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001),
                                   ny.arange(0, 1, 0.0001), ny.arange(0, 1, 0.0001)])

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = forwardprob(solution, U, Y, na, nt, ns, np)

print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

ga_instance.plot_fitness()

filename = '071922_INDPSC(250)'
ga_instance.save(filename=filename)
