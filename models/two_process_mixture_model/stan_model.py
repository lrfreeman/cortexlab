import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import time
import arviz as az

#Modify system pathway to ensure import works
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/models/mouse_task_GLM_parameters')

#Performance checks
start_time = time.time()

from GLM_classes import Data

model = """

// Stan code for a mixture-of-exponentials model on the distracer rewards task
data {

    int<lower=0> nTrials;
    // int<lower=0,upper=1> trial_types[nTrials]; - not needed for synethic data
    int<lower=-1,upper=1> choices[nTrials]; //
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
            H = (1- alpha_habits)* H + alpha_habits * choices[t_i];
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
for i in range(len(choices)):
    if choices[i] == 0:
        choices[i] = -1

nTrials = len(choices)

outcomes_grape = data["left_rewards"]
outcomes_grape = np.array(outcomes_grape, dtype=np.int)

outcomes_cherry = data["right_rewards"]
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
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)
print(fit)

#Create df for the sample data
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'],
                  columns=summary_dict['summary_colnames'],
                  index=summary_dict['summary_rownames'])

# Parameter estimation
# param_est = sm.optimizing(data=data)
# print(param_est)

#Extract the traces
u_cherry = fit['u_cherry']
u_grape = fit['u_grape']
u_nothing = fit['u_nothing']
beta_rl = fit['beta_rl']
beta_habits = fit['beta_habits']
beta_bias = fit['beta_bias']
alpha_rl = fit['alpha_rl']
alpha_habits = fit['alpha_habits']

def plot_posteriors(param, param_name='parameter'):
  """Plot the trace and posterior of a parameter."""

  # Summary statistics
  mean = np.mean(param)
  # median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

  # Plotting traces
  # plt.subplot(2,1,1)
  # plt.plot(param)
  # plt.xlabel('samples')
  # plt.ylabel(param_name)
  # plt.axhline(mean, color='r', lw=2, linestyle='--')
  # # plt.axhline(median, color='c', lw=2, linestyle='--')
  # plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  # plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  # plt.title('Trace and Posterior Distribution for {}'.format(param_name))

  #Plot Posterior Distributions
  plt.title('Posterior Distribution for {}'.format(param_name))
  plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
  plt.xlabel(param_name)
  plt.ylabel('density')
  plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  # plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
  plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  plt.gcf().tight_layout()
  # plt.legend()
  plt.show()

plot_posteriors(u_cherry, "u_cherry")
plot_posteriors(u_grape, "u_grape")
plot_posteriors(u_nothing, "u_nothing")
plot_posteriors(beta_rl, "beta_rl")
plot_posteriors(beta_habits, "beta_habits")
plot_posteriors(beta_bias, "beta_bias")
plot_posteriors(alpha_rl, "alpha_rl")
plot_posteriors(alpha_habits, "alpha_habits")

#Print the time of the process
print("")
print("--- %s seconds ---" % (time.time() - start_time))




print("")
