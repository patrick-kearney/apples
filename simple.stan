"""
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
  b0_bar ~ normal(70, 10);
  b1_bar ~ normal(-15, 10); 
  b2_bar ~ normal(0.031, 0.0043);
  
  
  for (apple in 1:N) {
    //defining each parameter as the sum of effects
    b0_apple[apple] = b0_bar;
    b1_apple[apple] = b1_bar;
    b2_apple[apple] = b2_bar;
    
    //growth curve for mean apple diameter over the season
    mu[apple] = b0_apple[apple]/(1+exp(-b2_apple[apple]*(day[apple] - b1_apple[apple])));
  
  }
  diameter ~ normal(mu, sigma);
  
}

"""
