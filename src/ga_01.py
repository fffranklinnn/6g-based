import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure


def run(problem, params):
    costfunc = problem.costfunc
    var_shape = problem.var_shape
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc * npop / 2) * 2)

    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = initialize(var_shape)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    bestcost = np.empty(maxit)

    for it in range(maxit):
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(-beta * costs)

        popc = []
        for _ in range(nc // 2):
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
            c1, c2 = crossover(p1, p2, var_shape)
            c1 = mutate(c1)
            c2 = mutate(c2)
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()
            popc.append(c1)
            popc.append(c2)

        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[:npop]
        bestcost[it] = bestsol.cost
        print(f"Iteration {it}: Best Cost = {bestcost[it]}")

    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out


def initialize(shape):
    position = np.zeros(shape, dtype=int)
    for i in range(shape[0]):
        position[i, np.random.choice(shape[1])] = 1
    return position


def crossover(p1, p2, shape):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    for i in range(shape[0]):
        if np.random.rand() < 0.5:
            c1.position[i] = p2.position[i]
            c2.position[i] = p1.position[i]
    return c1, c2


def mutate(x):
    y = x.deepcopy()
    for i in range(x.position.shape[0]):
        if np.random.rand() < 0.1:  # Mutation probability
            cols = np.where(y.position[i] == 1)[0]
            if len(cols) > 0:
                j = cols[0]
                new_j = np.random.choice([col for col in range(y.position.shape[1]) if col != j])
                y.position[i, j] = 0
                y.position[i, new_j] = 1
    return y


def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


# Sphere Test Function
def sphere(x):
    return np.sum(x ** 2)


# Problem Definition
problem = structure()
problem.costfunc = sphere
problem.var_shape = (10, 301)

# GA Parameters
params = structure()
params.maxit = 500
params.npop = 5
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
out = run(problem, params)

# Results
plt.plot(out.bestcost)
print(out.bestsol.position)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
