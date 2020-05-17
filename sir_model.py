import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

""" A modified version of the SIR model used in Homework 4"""

class SIR:
    def __init__(self, mu, beta, infected, population, n=1000, max_t=100):
        """
        Args:
            mu: Mu as defined in the basic SIR equations
            beta: Beta as defined in the basic SIR equations
            infected: number of people infected initially
            population: the total number of people represented in the curve
            n: number of time steps
            max_t: upper bound of number of days
        """
        self.mu = mu
        self.beta = beta
        self.population = float(population)
        self.n = n
        self.max_t = max_t
        I0 = float(infected)/population
        S0 = 1-I0
        R0 = 0
        self.t = np.linspace(0, max_t, n)
        self.I = np.empty_like(self.t)

        # Take initial time step (ie, I[0])
        self.I[0] = I0
        self.step = 1
        self.y0 = [S0, I0, R0]

    def sir_poisson(self, y, t, mu, b):
        """
        :param y: array of y-args
        :param t: timestamp
        :param mu: value from a mu-centered poisson distribution
        :param b: beta
        :return: array of derivatives of [s, i, r] at the timestamp
        """
        S = y[0]
        I = y[1]
        R = y[2]
        # Model equation
        dsdt = -b * I * S
        didt = b * I * S - mu * I
        drdt = mu * I
        return [dsdt, didt, drdt]

    def take_step(self):
        if self.step < self.n:
            tspan = [self.t[self.step-1], self.t[self.step]]
            sol = odeint(self.sir_poisson, self.y0, tspan, args=(float(np.random.poisson(self.mu*100)/100), self.beta))
            self.y0 = sol[1]
            self.I[self.step] = self.y0[1]
            self.step += 1

    def infect_other(self, destination, prob=0.05):
        """
        in @prob cases, this model will infect another model if:
            * this model has a current infected population (y0[1] > 0), and
            * @destination has a susceptible population (y0[0] > 0)
        :param destination: other model to potentially infect
        :param prob: probability that infecting the other model occurs
        :return: True if infection was successful, False otherwise
        """
        if self.y0[1] > 0 and destination.y0[0] > 0: # check if the two requirements are met
            num = random.random()
            if num <= prob:
                destination.y0[1] += 1.0/destination.population
                destination.y0[0] -= 1.0/destination.population
                return True
        return False

    def plot_sir(self):
        plt.plot(self.t, self.I, label='Infected')
        plt.title('SIR Model (mu={}, beta={}'.format(self.mu, self.beta))
        plt.xlabel('t')
        plt.ylabel('proportion of population infected')
        plt.legend()
        plt.show()
