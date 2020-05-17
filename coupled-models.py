from sir_model import SIR
import matplotlib.pyplot as plt
import numpy as np

""" 
Creating a coupled system of 25 SIR Models. Each SIR Model has the potential to infect other models. Only one initial 
model will contain an infected individual, to show a simple simulation of how a pandemic can spread through multiple
populations.

This is a simple "world" of countries with populations typically between 2 million and 8 million inhabitants
"""
# Initial defining of models
models = []
file = open('infections.txt', 'w')

for i in range(25):
    mu = max(0.01, np.random.normal(0.5, 0.15))
    beta = max(0.01, np.random.normal(0.5, 0.2))
    population = np.random.normal(5000000, 1000000)
    if len(models) == 0:
        models.append(SIR(mu, beta, 1, population, n=3000, max_t=300))
    else:
        models.append(SIR(mu, beta, 0, population, n=3000, max_t=300))

# Running the models for n timesteps
for i in range(3000): # default number of timesteps in sir_model class
    for j in range(25):
        models[j].take_step()
        other_model = int(np.random.uniform(0, 25))
        while other_model == j: # Ensures that the current model does not attempt to infect itself
            other_model = int(np.random.uniform(0, 25))
        if models[j].infect_other(models[other_model]):
            file.write("Day {}, Timestep {}: Model {} infects model {}\n".format(int(i/10), i % 10, j, other_model))
file.close()

# Plot results for all models
for i in range(25):
    plt.plot(models[i].t, models[i].I, label="Model {}".format(i))
plt.title("'World-Wide' SIR Model")
plt.xlabel("t")
plt.ylabel('proportion of population infected')
plt.legend()
plt.show()

# Allows user to pick single model of interest
model = input('Enter the model number to be plotted separately (or hit enter to end): ')
while True:
    if int(model) < 25:
        print(int(model))
        models[int(model)].plot_sir()
    model = input('Enter the model number to be plotted separately (or hit enter to end): ')