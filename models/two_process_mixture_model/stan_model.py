import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

#Modify system pathway to ensure import works
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/models/mouse_task_GLM_parameters')

from GLM_classes import Data

model = """

// Stan code for a mixture-of-exponentials model on the distracer rewards task
data {

    int<lower=0> nTrials;
    // int<lower=0,upper=1> trial_types[nTrials]; - not needed for synethic data
    int<lower=-1,upper=1> choices[nTrials];
    int<lower=0,upper=1> outcomes_cherry[nTrials];
    int<lower=0,upper=1> outcomes_grape[nTrials];

}
parameters {
    real u_cherry;
    real u_grape;
    real u_nothing;

    real<lower=0> beta_rl;
    real<lower=0> beta_habits;
    real beta_bias;

    real<lower=0, upper=1> alpha_rl;
    real<lower=0, upper=1> alpha_habits;
}
transformed parameters {
    real log_lik; // Accumulator for log-likelihood
    // Name-space for the loop over trials
    {
        real Q; // The hidden state of each agent
        real H; // The hidden state of each agent
        real Qeff;
        real u_trial;

        log_lik = 0;
        Q = 0;
        H = 0;

        for (t_i in 1:nTrials) {

            // if (trial_types[t_i] == 1)- needed for bev data not syth data
            Qeff = choices[t_i]*(beta_rl*Q + beta_habits*H + beta_bias);
            log_lik = log_lik + log(exp(Qeff) / (exp(Qeff) + exp(-1*Qeff)));

            // RL Learning
            if (outcomes_cherry[t_i] == 1 && outcomes_grape[t_i] == 1){
                u_trial = 1;
            }
            else if (outcomes_grape[t_i] == 1){
                u_trial = u_grape;
            }
            else if (outcomes_cherry[t_i] == 1){
                u_trial = u_cherry;
            }
            else {
                u_trial = u_nothing;
            }

            Q = (1-alpha_rl)*Q + alpha_rl * choices[t_i] * u_trial;
            // Habits learning
            H = (1-alpha_habits)*H + alpha_habits * choices[t_i];
        }
    }
}
model {
// Priors
u_cherry ~ normal(0, 1);
u_grape ~ normal(0, 1);
u_nothing ~ normal(0, 1);
beta_rl ~ normal(0, 1);
beta_bias ~ normal(0, 1);
alpha_rl ~ beta(3,3);
alpha_habits ~ beta(3,3);
// increment log likelihood
target += log_lik;
}
"""

#Load Data from a personal library that generated synthetic data
data = Data("s_data")._data_frame

#Map data from data frame to stan variables
choices = data["left_choices"]
choices = np.array(choices, dtype=np.int)

nTrials = len(choices)

outcomes_grape = data["right_rewards"]
outcomes_grape = np.array(outcomes_grape, dtype=np.int)

outcomes_cherry = data["left_rewards"]
outcomes_cherry = np.array(outcomes_cherry, dtype=np.int)

# Parameters to be inferred - Why put in parameters when these are the ones that get inferred?
u_cherry = 1
u_grape = 0
u_nothing = -1
beta_rl = 3
beta_habits = 1
beta_bias = 1
alpha_rl = 0.5
alpha_habits = 0.1

#Put the data in a dictionary
data = {'nTrials': nTrials,
        'choices': choices,
        'outcomes_grape': outcomes_grape,
        'outcomes_cherry': outcomes_cherry}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
# fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)
# # print(fit)

# Parameter estimation?
test = sm.optimizing(data=data)
print(test)
