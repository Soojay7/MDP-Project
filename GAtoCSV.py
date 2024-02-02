import numpy as ny
import pandas as ps
import MDPmodel as mdp
import csv

# This file is run to output optimized model and generation info in a CSV file format
Generations = mdp.Generations
Fitness = mdp.Fitness
best_solution = [l.tolist() for l in mdp.best_solution]


dictionary = {'Generations': Generations, 'Fitness': Fitness, 'Best Solution': best_solution}
df = ps.DataFrame(dictionary)
df = ps.concat([df['Generations'], df['Fitness'], ps.DataFrame(df['Best Solution'].to_list(), columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'])], axis=1)

ps.option_context("display.max_rows", None, "display.max_columns", None, 'display.width', None, 'display.width', None)
print(df)

df.to_csv('GAdata.csv', index=False)