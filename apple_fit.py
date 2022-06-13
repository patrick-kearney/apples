# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 22:51:00 2022

@author: Patrick
"""

# def main():
# get user input on model, variety, iteration/burn in
# and whether to fit to whole data
# or predict a season after how many weeks

# model = input("what model to run:  ")
# variety = input("what variety to run")


import stan
import pandas as pd
import arviz as az
import pickle
import os

model_type = 'simple'
variety = 'rg'
season = '7'


compiled_model_name = 'compiled_model.pic'
fit_save_name = model_type + '_' + variety + '_' + season + 'fit.pic'
raw_data = pd.read_csv('rg7.csv')
df = raw_data.loc[:, ['day', 'diameter']]
data = df.to_dict(orient='list')
data['N'] = df.shape[0]


def save(obj, filename):
    """save compiled models and fits"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    '''Reload pickled compiled models and fits for reuse'''
    return pickle.load(open(filename, 'rb'))


model = """
data {
  int<lower=0> N; //number of samples in data
  vector[N] diameter; 
  vector[N] day; //every diameter measurement has an associated week
}

// standard deviations must be positive

parameters {
  //average population parameter
  real<lower=0> b0_bar; // final average apple size always positive
  real b1_bar;  //season mid-point may be before or after end of calendar year, ie can be <0 or >0
  real<lower=0> b2_bar; //growth rate always positive
  
  // apple variance around logistic growth function
  //real<lower=0> sigma_sq;
  real<lower=0> sigma;
}

//transformed parameters {
//  real<lower=0> sigma; //standard deviation must be positive
//  sigma = sqrt(sigma_sq);
//}

model {
  vector[N] mu;
  vector[N] b0_apple;
  vector[N] b1_apple;
  vector[N] b2_apple;
  //sigma_sq ~ inv_gamma(0.001, 0.001);
  sigma ~ exponential(0.1);
  
  //priors on average parameter value
  b0_bar ~ normal(7, 1);
  b1_bar ~ normal(-1.5, 1); 
  b2_bar ~ normal(3.1, 0.43);
  
  
  for (apple in 1:N) {
    //defining each parameter as the sum of effects
    b0_apple[apple] = 10 * b0_bar;
    b1_apple[apple] = 10 * b1_bar;
    b2_apple[apple] = 0.01 * b2_bar;
    
    //growth curve for mean apple diameter over the season
    mu[apple] = b0_apple[apple]/(1+exp(-b2_apple[apple]*(day[apple] - b1_apple[apple])));
  
  }
  diameter ~ normal(mu, sigma);
  
}

"""

# if compiled model exists, don't need to compile
# else compile and pickle the compiled model for future runs

if exists(compiled_model_name):
    compiled_model = load(compiled_model_name)
else:
    compiled_model = stan.build(model, data=data)
    save(compiled_model, compiled_model_name)

# sample from compiled model
fit = compiled_model.sample(num_chains=4, num_samples=1000)

df = fit.to_frame()
print(df)
print(df.describe().T)

az.summary(fit)


save(fit, 'simple_model_rg_season_7_fit.pic')


my_fit = load('simple_model_rg_season_7_fit.pic')
az.summary(my_fit)






# import data from csv

# turn csv into dictionary

# select model


# start timer
# fit model
# end timer

# update log.txt with timer info etc


# pickle model
# save pickle to apples directory

# notify me by email that model has finished


# if __name__ == "__main__":
#    main()
