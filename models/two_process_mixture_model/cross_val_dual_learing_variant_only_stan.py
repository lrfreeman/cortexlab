import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import time
import arviz as az
import xarray as xr

#Modify system pathway to ensure import works
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/models/mouse_task_GLM_parameters')

#Performance checks
start_time = time.time()

from GLM_classes import Data

model = """

// Stan code for a mixture-of-exponentials model on the distracer rewards task

data {

    //Training Data variables
    int<lower=0> nTrials;
    int<lower=-1,upper=1> choices[nTrials];
    int<lower=0,upper=1> outcomes_cherry[nTrials];
    int<lower=0,upper=1> outcomes_grape[nTrials];

    //Testing Data variables
    int<lower=0> nTrials_test;
    int<lower=-1,upper=1> choices_test[nTrials_test];
    int<lower=0,upper=1> outcomes_cherry_test[nTrials_test];
    int<lower=0,upper=1> outcomes_grape_test[nTrials_test];

}

parameters {
    real u_cherry;
    real u_grape;
    real u_nothing;

    real<lower=0> beta_rl;
    real<lower=0> beta_habits;
    real beta_bias;

    real<lower=0, upper=1> positive_alpha_rl;
    real<lower=0, upper=1> negative_alpha_rl;
    real<lower=0, upper=1> alpha_habits;
}

transformed parameters {
    real log_lik; // Accumulator for log-likelihood

    // Name-space for the loop over trials
    {

        real Q; // The hidden state of each agent
        real H; // The hidden state of each agent
        real Qeff; // q effective goes into softmax / logit (weighted sum of q and h)
        real u_trial;

        log_lik = 0;
        Q = 0;
        H = 0;

        for (t_i in 1:nTrials) {

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

            if ((u_trial - Q) > 0) {
                Q = (1-positive_alpha_rl) * Q + positive_alpha_rl * choices[t_i] * u_trial;
            }

            else if ((u_trial - Q) < 0) {
                Q = (1-negative_alpha_rl) * Q + negative_alpha_rl * choices[t_i] * u_trial;

            }

            // Habits learning
            H = (1-alpha_habits) * H + alpha_habits * choices[t_i];
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
positive_alpha_rl ~ beta(3,3);
negative_alpha_rl ~ beta(3,3);
alpha_habits ~ beta(3,3);

// increment log likelihood
target += log_lik;
}

generated quantities {
    real log_lik_test; // Accumulator for log-likelihood
    // Name-space for the loop over trials
    {
        real Q; // The hidden state of each agent
        real H; // The hidden state of each agent
        real Qeff;
        real u_trial;

        log_lik_test = 0;
        Q = 0;
        H = 0;

        for (t_i in 1:nTrials_test) {

            Qeff = choices_test[t_i]*(beta_rl*Q + beta_habits*H + beta_bias);

            log_lik_test = log_lik_test + log(exp(Qeff) / (exp(Qeff) + exp(-1*Qeff)));

            // RL Learning
            if (outcomes_cherry_test[t_i] == 1 && outcomes_grape_test[t_i] == 1){
                u_trial = 1;
            }
            else if (outcomes_grape_test[t_i] == 1){
                u_trial = u_grape;
            }
            else if (outcomes_cherry_test[t_i] == 1){
                u_trial = u_cherry;
            }
            else {
                u_trial = u_nothing;
            }

            if ((u_trial - Q) > 0) {
                Q = (1-positive_alpha_rl) * Q + positive_alpha_rl * choices[t_i] * u_trial;
            }

            else if ((u_trial - Q) < 0) {
                Q = (1-negative_alpha_rl) * Q + negative_alpha_rl * choices[t_i] * u_trial;

            }

            // Habits learning
            H = (1-alpha_habits) * H + alpha_habits * choices_test[t_i];
        }
    }
}
"""

#Load Data from a personal library that generated synthetic data
training_data = Data("train_b_data")._data_frame

#####Training Code##########
#Map data from data frame to stan variables
choices = training_data["left_choices"]
choices = np.array(choices, dtype=np.int)
for i in range(len(choices)):
    if choices[i] == 0:
        choices[i] = -1
nTrials = len(choices)
outcomes_grape = training_data["left_rewards"]
outcomes_grape = np.array(outcomes_grape, dtype=np.int)
outcomes_cherry = training_data["right_rewards"]
outcomes_cherry = np.array(outcomes_cherry, dtype=np.int)

##Test Code
test_data = Data("test_b_data")._data_frame
#####Training Code##########
#Map data from data frame to stan variables
choices_test = test_data["left_choices"]
choices_test = np.array(choices_test, dtype=np.int)
for i in range(len(choices_test)):
    if choices_test[i] == 0:
        choices_test[i] = -1
nTrials_test = len(choices_test)
outcomes_grape_test = test_data["left_rewards"]
outcomes_grape_test = np.array(outcomes_grape_test, dtype=np.int)
outcomes_cherry_test = test_data["right_rewards"]
outcomes_cherry_test = np.array(outcomes_cherry_test, dtype=np.int)

#Put the data in a dictionary
test_dic = {'nTrials': nTrials,
            'choices': choices,
            'outcomes_grape': outcomes_grape,
            'outcomes_cherry': outcomes_cherry,
            'nTrials_test': nTrials_test,
            'choices_test': choices_test,
            'outcomes_grape_test': outcomes_grape_test,
            'outcomes_cherry_test': outcomes_cherry_test
}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Parameter estimation
test_param_est = sm.optimizing(data=test_dic)

#Output LogLikihood
print("Output of cross validation:")
print("test LL: ", np.exp(test_param_est["log_lik_test"]/nTrials_test))
print("train LL:", np.exp(test_param_est["log_lik"]/nTrials))

#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))
print("")
