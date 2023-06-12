#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:02:13 2022

@author: w044elc
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(GaussianMLP, self).__init__()

        # first half outputs mean value, last half outputs variance
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, int(hidden_size), bias=True)
        nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(int(hidden_size), int(hidden_size), bias=True)
        nn.init.xavier_normal_(self.fc2.weight)

        self.output_layer = nn.Linear(hidden_size, 2*self.output_dim, bias=True)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, input_states):
        input_states = input_states.float()
        x = self.fc1(input_states)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        output = self.output_layer(x)

        return output

    def get_mean_std(self, x):
        with torch.no_grad():
            output = self.forward(x)

            if output.dim() == 2:
                mean = output[:, :self.output_dim]
                logvar = output[:, self.output_dim:]
            else:
                mean = output[:self.output_dim]
                logvar = output[self.output_dim:]

        return mean, 0.5*logvar

    def get_distribution(self, x):
        with torch.no_grad():
            output = self.forward(x)

            if output.dim() == 2:
                mean = output[:, :self.output_dim]
                logvar = output[:, self.output_dim:]
            else:
                mean = output[:self.output_dim]
                logvar = output[self.output_dim:]                

            # prohibit var becoming 0
            var = torch.exp(logvar) + 0.001 * torch.ones_like(logvar)

            gaussian_distribution = []
            
            if output.dim() == 2:
                for i in range(output.shape[0]):
                    gaussian_distribution.append(torch.distributions.normal.Normal(mean[i,:], var[i,:]))
            else:
                gaussian_distribution.append(torch.distributions.normal.Normal(mean, var))

        return gaussian_distribution


class PNNLoss_Gaussian(nn.Module):
    '''
    Here is a brief aside on why we want and will use this loss. Essentially, we will incorporate this loss function to include a probablistic nature to the dynamics learning nueral nets. The output of the Probablistic Nueral Net (PNN) or Bayesian Neural Net (BNN) will be both a mean for each trained variable and an associated variance. This loss function will take the mean (u), variance (sig), AND the true trained value (s) to compare against the mean. Stacked variances form Cov matrix
    loss_gaussian = sum_{data} (u - s)^T Cov^-1 (u-s) + log Det(Cov)
    Need to add code like this to the implementation:
         To bound the variance output for a probabilistic network to be between the upper and lower bounds found during training the network on the training data, we used the following code with automatic differentiation:
         logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
         logvar = min_logvar + tf.nn.softplus(logvar - min_logvar)
         var = torch.exp(logvar)
         with a small regularization penalty on term on max_logvar so that it does not grow beyond the training distribution’s maximum output variance, and on the negative of min_logvar so that it does not drop below the training distribution’s minimum output variance.
    '''

    def __init__(self, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        super(PNNLoss_Gaussian, self).__init__()

        self.idx = idx
        self.initialized_maxmin_logvar = True
        # Scalars are proportional to the variance to the loaded prediction data
        # self.scalers    = torch.tensor([2.81690141, 2.81690141, 1.0, 0.02749491, 0.02615976, 0.00791358])
        self.scalers = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])

        # weight the parts of loss
        self.lambda_cov = 1  # scaling the log(cov()) term in loss function
        self.lambda_mean = 1

    def set_lambdas(self, l_mean, l_cov):
        # sets the weights of the loss function
        self.lambda_cov = l_mean
        self.lambda_mean = l_cov

    def get_datascaler(self):
        return self.scalers

    def softplus_raw(self, input):
        # Performs the elementwise softplus on the input
        # softplus(x) = 1/B * log(1+exp(B*x))
        B = torch.tensor(1, dtype=torch.float)
        return (torch.log(1 + torch.exp(input.mul_(B)))).div_(B)

    def forward(self, output, target, max_logvar, min_logvar):
        '''
        output is a vector of length 2d
        mean is a vector of length d, which is the first set of outputs of the PNN
        var is a vector of variances for each of the respective means
        target is a vector of the target values for each of the mean
        '''

        # Initializes parameterss
        d2 = output.size()[1]
        d = torch.tensor(d2 / 2, dtype=torch.int32)
        mean = output[:, :d]
        
        logvar = output[:, d:]
        # Caps max and min log to avoid NaNs
        logvar = max_logvar - self.softplus_raw(max_logvar - logvar)
        logvar = min_logvar + self.softplus_raw(logvar - min_logvar)

        # Computes loss
        var = torch.exp(logvar)
        b_s = mean.size()[0]  # batch size

        eps = 0  # Add to variance to avoid 1/0

        # A = mean - target.expand_as(mean)
        # A.mul_(self.scalers)
        # B = torch.div(mean - target.expand_as(mean), var.add(eps))
        # # B.mul_(self.scalers)
        # loss = torch.sum(self.lambda_mean * torch.bmm(A.view(b_s, 1, -1), B.view(b_s, -1, 1)).reshape(-1,1) + self.lambda_cov * torch.log(torch.abs(torch.prod(var.add(eps), 1)).reshape(-1, 1)))

        A = self.lambda_mean * torch.sum(((mean - target.expand_as(mean)) ** 2) / var, 1)

        B = self.lambda_cov * torch.log(torch.abs(torch.prod(var.add(eps), 1)))

        loss = torch.sum(A + B)

        return loss
    

class Regressor(): 
    def __init__(self, input_dim, output_dim, hidden_sizes): 
        
        self.model = GaussianMLP(input_dim, output_dim, hidden_sizes)
        self.loss = PNNLoss_Gaussian()
        
    def fit(self, x_inp, out_real, epochs = 50): 
        optimizer = torch.optim.Adam(params = self.model.parameters(), lr = 1e-4)
        
        for j in range(epochs): 
            out = self.model(x_inp)
            d2 = out.size()[1]
            d = torch.tensor(d2 / 2, dtype=torch.int32)
            log_var = out[:,d:]
            max_indx = torch.argmax(log_var, dim = 1, keepdim = True)
            min_indx = torch.argmin(log_var, dim = 1, keepdim = True)
            
            max_logvar = torch.zeros([log_var.shape[0], 1])
            min_logvar = torch.zeros([log_var.shape[0], 1])
            for i in range(log_var.shape[0]):
                max_logvar[i] = log_var[i, max_indx[i]]
                min_logvar[i] = log_var[i, min_indx[i]]
                
            loss = self.loss(out, out_real, max_logvar, min_logvar)
            loss.backward()
            optimizer.step()
            
    def predict(self, x_inp): 
        prediction = self.model(x_inp)
        return prediction 
    
    def log_likelihood(self, x_inp, y): 
        mean, logstd = self.model.get_mean_std(x_inp)
        z = (y - mean) / np.exp(logstd)
        log_likelihood = - torch.sum(logstd, dim = 1, keepdim = True) - \
                            0.5 * torch.sum(np.square(z), dim = 1, keepdim = True) - \
                            0.5 * mean.shape[-1] * np.log(2 * np.pi)
        return log_likelihood
