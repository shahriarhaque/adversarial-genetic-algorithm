# GA Reverse Engineer

import array
import random

import numpy
import simutils

from deap import algorithms
from deap import base
from deap import creator
from deap import tools



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_int", random.randint, 0, simutils.NUM_COMMON_WORDS-1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, simutils.MAX_WORDS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    # return sum(individual),
    return (1.0 - simutils.fitness(individual)),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.5, ngen=15,
                                   stats=stats, halloffame=hof, verbose=True)

    for i in range(5):
        best = simutils.individual_to_text(hof[i])
        print(best)
        print(simutils.fitness_text(best))

if __name__ == "__main__":
    main()
