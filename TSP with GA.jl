## Problem: Traveling salesman problem, objective function: minimize total distance traveled
## Solver: Genetic algorithm
## Language: Julia
## Written by: @setyotw
## Purpose: Public repository
## Date: September 5, 2022


#%% import packages
using Pkg, Random, DataFrames
Pkg.status()

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS

#%% define functions needed
# a | creating an initial population
function initializePopulation(popSize, numNodes)
    population = ones(Int, popSize, numNodes+1)
    for i in range(1, popSize)
        population[i, 2:end-1] = randperm(numNodes-1).+1
    end
    return population
end

# b | calculate the travel cost (distance) of a single chromosome (solution) --> this is basically fitness function (inv)
function travelDistance(solution, distanceMatrix)
    distance = float(0)
    distance = distanceMatrix[1,solution[2]] + distanceMatrix[solution[end-1],1]
    departureNode = solution[2]
    for i in range(2,length(solution)-1)
        currentNode = solution[i]
        distance = distance + distanceMatrix[departureNode, currentNode]
        departureNode = solution[i]
    end
    return distance
end

# c | calculate the fitness of a population
function fitnessCalculation(population, popSize, distanceMatrix)
    fitnessPop = zeros(popSize,1)
    for i in range(1,popSize)
        solution = population[i,1:end]
        solutionDistance = travelDistance(solution, distanceMatrix)
        fitnessPop[i] = 1/solutionDistance
    end
    return fitnessPop
end

# d | rank the solution chromosomes
function rankSolution(population, fitnessPop, popSize)
    rankIndex = sortperm(vec(fitnessPop), rev=true) # rev=true --> descending
    rankIndex = rankIndex[1:popSize]
    population = population[rankIndex,:]
    fitnessPop = fitnessPop[rankIndex,:]
    return population, fitnessPop
end

# e | roulette wheel for mating population
function rouletteWheel(population, fitnessPop, poolSize, numNodes)
    fitnessCumSum = cumsum(fitnessPop; dims=1)
    fitnessSum = sum(fitnessPop)
    selectionChance = 100*fitnessCumSum/fitnessSum
    indexPool = zeros(Int, poolSize)
    parentPool = ones(Int, poolSize, numNodes+1)
    for i in range(1, poolSize)
        r = 100*rand(1)
        indexPool[i] = Int(length(selectionChance[r.>selectionChance])+1)
        parentPool[i,:] = population[indexPool[i],:]
    end
    return parentPool
end

# f | breeding function to create new offsprings
function breeding(crossoverRate, parentPool, popSize, poolSize, numNodes)
    childrens = ones(Int, popSize, numNodes+1)
    i = Int(1)
    while i < popSize
        parentIndex = randperm(poolSize)[1:2]
        parent_1 = parentPool[parentIndex[1],:]
        parent_2 = parentPool[parentIndex[2],:]
        if rand(1)[1] <= crossoverRate
            childrens[i,:], childrens[i+1,:] = singlePointCrossover(parent_1, parent_2, numNodes)
        else
            childrens[i,:] = parent_1
            childrens[i+1,:] = parent_2
        end
        i = i+2
    end
    return childrens
end

# g | single-point crossover
function singlePointCrossover(parent_1, parent_2, numNodes)
    childs = ones(Int, 2, numNodes+1)
    parents = (parent_1, parent_2)
    swappingPoint = (randperm(numNodes-1).+1)[1]
    # swap them
    childs[1,:] = vcat(parents[1][1:swappingPoint], parents[2][swappingPoint+1:end])
    childs[2,:] = vcat(parents[2][1:swappingPoint], parents[1][swappingPoint+1:end])

    # correct the duplications
    for i in range(1,2)
        childOtherItem = [item for item in parents[i][swappingPoint+1:end] if item ∉ childs[i,swappingPoint+1:end]] # find nodes that not appear at the child
        duplicateItems = [item for item in childs[i,swappingPoint+1:end-1] if item ∈ childs[i,1:swappingPoint]]  # find the duplicated indexes
        if isempty(childOtherItem) == true
            continue
        else
            duplicateIndex = [findall(x -> x==duplicateItems[j], childs[i,swappingPoint+1:end-1])[1]+swappingPoint for j in range(1, length(duplicateItems))]
            childs[i,duplicateIndex] = childOtherItem               
        end
    end

    return childs[1,:], childs[2,:]
end

# h | swap mutation
function mutatePop(population, popSize, numNodes, mutationRate)
    for i in range(1, popSize)
        if rand(1)[1] <= mutationRate
            points = (randperm(numNodes-1).+1)[1:2]
            pointA = minimum(points)
            pointB = maximum(points)
            mutableA = population[i, pointA]
            mutableB = population[i, pointB]
            population[i, pointA] = mutableB
            population[i, pointB] = mutableA
        end
    end
    return population
end

# i | GA framework
function geneticAlgorithm(popSize, crossoverRate, mutationRate, iterationNum, numNodes, distanceMatrix)
    # 1 | create the initial population
    # [1, 2, 6, 3, 5, 4, 1]
    initialPop = initializePopulation(popSize, numNodes)
    # 2 | set it as the current population
    currentPop = initialPop    
    # 3 | calculate the fitness value of each chromosome
    currentFitness = fitnessCalculation(currentPop, popSize, distanceMatrix)
    # 4 | rank the chromosome based on its fitness value
    currentPop, currentFitness = rankSolution(currentPop, currentFitness, popSize)

    # 5 | for each iteration
    for i in range(1, iterationNum)
        # 6 | create a mating pool using roulette wheel selection
        poolSize = Int(floor(popSize/2))
        parentPool = rouletteWheel(currentPop, currentFitness, poolSize, numNodes)
        # 7 | genetic operator: breeding to create 'popSize' amount of new offsprings
        # using a single-point crossover
        childrens = breeding(crossoverRate, parentPool, popSize, poolSize, numNodes)
        # 8 | genetic operator: mutation to the parent population
        currentPop = mutatePop(currentPop, popSize, numNodes, mutationRate)
        # 9 | merge the currentPop with the new childrens
        mergedPop = vcat(currentPop, childrens)
        # 10 | calculate the fitness of the merged population
        mergedFitness = fitnessCalculation(mergedPop, popSize*2, distanceMatrix)
        # 11 | sort and cut the size into the original popsize
        currentPop, currentFitness = rankSolution(mergedPop, mergedFitness, popSize)
    end

    # 12 | extract the best chromosome (the 1st chromosome on the current population) and its objective value
    bestSol = currentPop[1,:]
    bestObjective = travelDistance(bestSol, distanceMatrix)

    return currentPop, currentFitness, bestSol, bestObjective
end
#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple TSP case with 1 depot and 10 customer nodes

# symmetric distance matrix [11 x 11]
distanceMatrix = Array{Float64}([
    0.000 2.768 7.525 9.689 28.045 36.075 25.754 3.713 2.701 8.286 7.944;
    2.768 0.000 9.164 11.482 27.779 37.431 25.488 7.305 1.928 7.911 10.758;
    7.525 9.164 0.000 16.297 33.406 42.246 31.115 4.065 8.062 13.970 1.594;
    9.689 11.482 16.297 0.000 19.053 27.198 16.762 12.484 8.810 17.182 20.464;
    28.045 27.779 33.406 19.053 0.000 10.138 5.596 31.349 26.286 36.994 38.820;
    36.075 37.431 42.246 27.198 10.138 0.000 10.634 38.434 34.759 43.132 46.850;
    25.754 25.488 31.115 16.762 5.596 10.634 0.000 29.058 23.995 32.498 36.529;
    3.713 7.305 4.065 12.484 31.349 38.434 29.058 0.000 6.005 8.615 5.651;
    2.701 1.928 8.062 8.810 26.286 34.759 23.995 6.005 0.000 9.862 9.656;
    8.286 7.911 13.970 17.182 36.994 43.132 32.498 8.615 9.862 0.000 15.313;
    7.944 10.758 1.594 20.464 38.820 46.850 36.529 5.651 9.656 15.313 0.000])

# number of nodes on the graph, can be calculated as the horizontal/vertical length of the distance matrix
numNodes = length(distanceMatrix[:,1])

#%% define parameters for the genetic algorithm
popSize = Int(10) # population size
poolSize = Int(popSize/2) # size of the mating pool
crossoverRate = float(0.7) # probability to execute crossover
mutationRate = float(0.3) # probability to execute mutation
iterationNum = Int(1000) # maximum number of iteration

# implement the GA to solve TSP
result_population, result_fitnessPop, result_bestPop, result_bestObjective = geneticAlgorithm(popSize, crossoverRate, mutationRate, iterationNum, numNodes, distanceMatrix)
