import numpy as ny
from numpy import random
import math
import pygad
from FwdProbFunc import *

# Genetic Algorithm
def ga_func(U, Y, na, nt, ns, np):

    # Defining fitness function for GA
    def fitness_func(solution, solution_idx):
        fitness = forwardprob(solution, U, Y, na, nt, ns, np)
        return fitness


    # Creating population of solutions
    num_solutions = 100 # 500
    pop_vector_int = ny.zeros(shape=(num_solutions,17))

    for solution_idx in range(num_solutions):
        r = ny.random.uniform(low=0, high=1, size=(17))
        pop_vector_int[solution_idx,:] = r


    # GA parameters
    num_generations = 300
    num_parents_mating = 50

    sol_per_pop = num_solutions
    pop_size = num_solutions
    num_genes = 17   # same as number of inputs

    init_range_low = 0
    init_range_high = 1

    parent_selection_type = "sss"
    keep_parents = -1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10


    # GA function for printing generation number and fitness after each generation iterated and storing best solutions to output in a CSV file
    best_solution = [] * num_generations
    def callback_gen(ga_instance):
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

    return ga_instance.best_solution()[0]