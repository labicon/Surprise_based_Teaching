import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.replay_buffer import ReplayBuffer

class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(GaussianMLP, self).__init__()

        # first half outputs mean value, last half outputs variance
        self.output_dim = output_dim
        self.min_logvar = -2 * torch.ones(output_dim)
        self.max_logvar = 2 * torch.ones(output_dim)

        self.fc1 = nn.Linear(input_dim, int(hidden_size), bias=True)
        nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(int(hidden_size), int(hidden_size), bias=True)
        nn.init.xavier_normal_(self.fc2.weight)

        self.fc3 = nn.Linear(int(hidden_size), int(hidden_size), bias=True)
        nn.init.xavier_normal_(self.fc3.weight)

        self.output_layer = nn.Linear(hidden_size, 2*self.output_dim, bias=True)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, input_states):
        input_states = input_states.float()
        x = self.fc1(input_states)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
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

            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

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

    def forward(self, mean, logvar, targets):
        '''
        output is a vector of length 2d
        mean is a vector of length d, which is the first set of outputs of the PNN
        var is a vector of variances for each of the respective means
        target is a vector of the target values for each of the mean
        '''

        inv_var = torch.exp(-logvar)

        mse_losses = torch.mean(torch.mean(torch.square(mean - targets) * inv_var, dim=-1), dim=-1)
        var_losses = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
        total_loss = mse_losses + var_losses

        return total_loss
    

class Regressor(): 
    def __init__(self, input_dim, output_dim, hidden_sizes, epochs = 1, batch_size = 128): 
        
        self.model = GaussianMLP(input_dim, output_dim, hidden_sizes)
        self.loss = PNNLoss_Gaussian()
        self.replay_buffer = ReplayBuffer()

        self.max_logvar = 2 * torch.ones(output_dim)
        self.min_logvar = -2 * torch.ones(output_dim)

        self.output_dim = output_dim
        
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x_inp, out_real): 
        optimizer = torch.optim.Adam(params = self.model.parameters(), lr = 1e-4)

        self.replay_buffer.add(x_inp, out_real)
        
        for j in range(self.epochs): 
            state_actions, next_states = self.replay_buffer.sample(batch_size = self.batch_size)

            predicted_next_states = self.model(state_actions)

            # process the output of the model
            mean = predicted_next_states[:,:self.output_dim]
            log_var = predicted_next_states[:,self.output_dim:]

            log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
            log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)
            
            loss = self.loss(mean, log_var, next_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def predict(self, x_inp): 
        prediction = self.model(x_inp)
        return prediction 
    
    def sample(self, x_inp): 
        mean, logstd = self.model.get_mean_std(x_inp)
        z = torch.randn(mean.shape)
        return mean + z * torch.exp(logstd)
    
    def log_likelihood(self, x_inp, y): 
        mean, logstd = self.model.get_mean_std(x_inp)
        z = (y - mean) / np.exp(logstd)
        log_likelihood = - torch.sum(logstd, dim = 1, keepdim = True) - \
                            0.5 * torch.sum(np.square(z), dim = 1, keepdim = True) - \
                            0.5 * mean.shape[-1] * np.log(2 * np.pi)
        return log_likelihood
