# import pickle fit
# unpickle
# extract values
# plot posterior predictive dist
# compare to actual data


import stan
import pandas as pd
import arviz as az
import pickle
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data = pd.read_csv('rg7.csv')



def load(filename):
    '''Reload pickled compiled models and fits for reuse'''
    return pickle.load(open(filename, 'rb'))


fit = load('simple_model_rg_season_7_fit.pic')

az.summary(fit)


df = fit.to_frame()


def generate_post_predictive(b0, b1, b2, sigma, t):
    """
    Return a diameter measurement for given parameter values at
    a given time
    """

    mu = b0/(1 + exp(-b2*(t - b1)))
    y = np.random.normal(loc=mu, scale=sigma)
    return y


#get time points from rg7
t_points = data.day.unique()
t_points.sort()


#get average for each time point
y_data = data.groupby(['day']).diameter.mean()
block_data = data.groupby(['day']).b_id.unique()
#plot average size over time

x = [1, 2, 3, 4, 5, 6, 10, 20]
y = [1, 4, 9, 16, 25, 36, 100, 400]
plt.plot(x, y)
#plt.show()
plt.savefig('rg7_average_data.png')
img = Image.open('rg7_average_data.png')
img.show()


