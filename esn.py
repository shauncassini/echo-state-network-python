import numpy as np
from scipy.linalg import diagsvd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from typing import Optional


class ESN():

    ''' An ESN class for conducting experiments

    Parameters (See init signature for default values and datatypes)
    ---------- Ambiguous parameters are described here

    data             -> A (n_timesteps x m_dimensions) time series
    snr              -> The signal-to-noise ratio for the time series
    rng              -> The RandomState (ensure this is set with each parameter change!)

    transient_time   -> 'Washout' time
    input_scaling    -> A scaling factor for the input weights
    (X)_distribution -> The distribution from ['uniform', 'normal'] to use for weight matrix ('W_in','R','W_fb')
    reg              -> Regularlization factor

    disable_tqdm     -> Set to False to disable progressbars - essential for parameter search
    preserve_state   -> If set to True, each training run gives the same result
    fast_solve       -> Use 'fast' linalg solving (slow for large N?)

    '''

    def __init__(

        self,

        data: np.ndarray,
        snr: float = 0.0,

        training_time: int = 900,
        resevoir_size: int = 500,
        transient_time: int = 20,
        transient_time_portion: float = None,
        
        leaking_rate: float = 0.3,
        sparsity: float = 0.3,
        input_scaling: float = 1.0,

        spectral_radius: float = 0.75,
        reg: float = 0,

        W_in_distribution: str = 'normal',
        R_distribution: str = 'normal',
        W_fb_distribution: str = 'normal',

        disable_tqdm: bool = True,
        preserve_state: bool = True,
        rng: np.random.RandomState = None,
        fast_solve: bool = False

    ):

        # Check correct distributions
        distributions = [
            W_in_distribution,
            R_distribution,
            W_fb_distribution
        ]
        for dist in distributions:
            if not dist in ['uniform','normal']:
                raise ValueError('Distributions either "uniform" or "normal"')

        if rng: 
            self.rng = rng
        else: 
            self.rng = np.random.RandomState(69420)

        if preserve_state: # Store the random state at initialization
            self.initial_state = self.rng.get_state()
        
        self.init_data = data.copy() # Copy original data to then modify with snr

        if snr > 0:
            self.init_snr = -1
        else:
            self.init_snr = snr
        
        self.data = data
        self.snr = snr

        self.k_dimensions  = data.shape[1]

        self.training_time = int(training_time)
        self.resevoir_size = int(resevoir_size)

        if transient_time_portion:
            self.transient_time = int(training_time * transient_time_portion)
        else:
            self.transient_time = int(transient_time)
        
        self.leaking_rate = leaking_rate
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.reg = reg

        self.W_in_distribution = W_in_distribution
        self.W_fb_distribution = W_fb_distribution
        self.R_distribution = R_distribution

        self.disable_tqdm = disable_tqdm
        self.preserve_state = preserve_state
        self.fast_solve = fast_solve

        self.trained = False


    def __add_noise(
        self, data: np.ndarray, snr: float
    ) -> np.ndarray:
        ''' Adds noise to data based on snr'''

        n = data.size
        # Add scaled noise based on signal-to-noise-ratio (SNR)
        
        noise = self.rng.normal(0, 1, n)[:, None]
        k = np.linalg.norm(data)/(np.linalg.norm(noise) * snr)
        data += noise * k

        return data


    def init_esn(self) -> None:
        ''' Initialises the ESN '''

        # Keeps the state the same at each initialization
        if self.preserve_state: 
            self.rng.set_state(self.initial_state)
        
        # Add noise to data if snr has changed
        if self.snr != self.init_snr:
            self.data = self.__add_noise(self.init_data, self.snr)
            self.init_snr = self.snr

        # Target data
        self.y = self.data[self.transient_time+1:self.training_time+1] # Predict from xi, predict xi+1

        #TODO: Only modify weights if weight parameters have changed
        self.__set_weights()
        self.__set_resevoir()

    
    def __set_weights(self) -> None:
        ''' Initialize the weights (in place) '''

        #'+1' is the bias term
        self.W_in = self.__get_dist(
            self.W_in_distribution, self.resevoir_size, self.k_dimensions + 1
        ) * self.input_scaling

        self.R = self.__get_dist(
            self.R_distribution, self.resevoir_size, self.resevoir_size
        )

        # Resevoir collection matrix (or 'design' matrix) where A[i] = [1, u[i], x[i]]
        self.A = np.zeros((
            self.training_time - self.transient_time, 
            1 + self.k_dimensions + self.resevoir_size
        ))

        self.W_out = np.zeros((
            self.resevoir_size + self.k_dimensions + 1, 
            self.k_dimensions
        ))


    def __get_dist(self, dist: str, n: int, m: int) -> np.ndarray:
        ''' Initialize (n x m) matrix from specified distribution

        Parameters
        ----------
        dist: str
            The type of distribution (defined at __init__)
        n, m: int 
            The dimensions of the matrix

        Returns
        ---------
            An (n x m) matrix sampled from a specified distribution

        '''

        if dist == 'uniform':
            return self.rng.uniform(-1, 1, size=(n, m))
        if dist == 'normal':
            return self.rng.randn(n, m)


    def __set_resevoir(self) -> None:
        ''' Configure the resevoir (in place) '''

        ### SPARSITY ###

        sparsity_ = int(self.sparsity * self.resevoir_size ** 2)
        indxs: np.ndarray = np.ones((self.resevoir_size ** 2))
        choice = self.rng.choice(range(self.resevoir_size ** 2), sparsity_)
        indxs[choice] = 0
        self.R *= indxs.reshape((self.resevoir_size, self.resevoir_size))

        ### SPECTRAL RADIUS ###

        eigvals = np.linalg.eig(self.R)[0]
        max_eigval = np.abs(eigvals).max()
        self.R /= self.spectral_radius * max_eigval


    def train(self) -> None:
        ''' Trains the ESN '''

        self.init_esn()

        # Collect resevoir states

        U = np.ones((2, self.training_time, 1))
        U[1, :] = self.data[:self.training_time]
 
        x = np.zeros((self.resevoir_size, 1))

        for i in tqdm(range(self.training_time), disable=self.disable_tqdm):

            u_in = U[:, i]

            x_leak = (1 - self.leaking_rate) * x
            x = x_leak + self.leaking_rate * np.tanh(self.W_in @ u_in + self.R @ x)

            if i > self.transient_time - 1:
                self.A[i-self.transient_time, 2:] = x.squeeze(-1)

        # Store the latest activation (for prediction)
        self.x_prev = x

        self.A[:,0] = np.ones(self.training_time - self.transient_time)
        self.A[:,1] = self.data[self.transient_time:self.training_time].ravel()

        # Solve for W_out in the form A @ W_out = y : Ax = b
        if self.fast_solve:
            self.W_out = self.__fast_linalg_solve(self.A, self.y, self.reg)
        else:
            self.__slow_linalg_solve()

        self.trained = True

    
    def __fast_linalg_solve(
        self, A: np.ndarray, b: np.ndarray, reg: float
    ) -> np.ndarray:
        ''' Fast method for solving Ax = b using SVD on A (thanks Joab) '''

        U, s, V_t = np.linalg.svd(A)
        S = diagsvd(s, A.shape[0], A.shape[1])
        I = reg * np.eye(S.shape[1])
        
        # Prevents inverse errors
        try:
            S_inv = np.linalg.inv(S.T @ S + I)
        except:
            print('Failed to invert array, using pseudo-inverse')
            S_inv = np.linalg.pinv(S.T @ S + I)
            
        S_1 = S.T @ U.T @ b
        x = V_t.T @ S_inv @ S_1

        return x

    
    def __slow_linalg_solve(self) -> None:
        ''' Older method for solving Ax = b (in place)'''

        I = np.eye(1 + self.k_dimensions + self.resevoir_size)
        # I[0,0] = 0 # To preseve bias

        # if self.reg > 0:
        #     A_inv = np.linalg.pinv(self.A.T @ self.A + self.reg * I)
        #     self.A = A_inv @ self.A.T
        # else:
        X = self.A.T @ self.A + self.reg * I
        A_inv = np.linalg.inv(X)
        self.W_out = A_inv @ (self.A.T @ self.y)


    def predict(
        self, predicted_steps: int, return_predicted: bool = False, plot: bool = False
    ) -> Tuple[Optional[np.ndarray], float]:
    
        '''Predicts next points in the data
            (Trains the ESN first if not yet trained)

        Parameters
        ----------
        predicted_steps: int
            How many timesteps to predict
        return_predicted: bool (Optional)
            Specify whether to return the data or not (default is False)
        plot: bool (Optional)
            Plot the predicted vs. true data along with its error (default is False)
        
        Returns
        ----------
        y_pred: np.ndarray(predicted_steps, 1)
            Predicted data
        rse: float
            The rse of the prediction
        '''

        # pre-allocate arrays instead of vstack (for efficiency)
        y_pred = np.zeros((predicted_steps, self.k_dimensions))
        y_pred = np.hstack((np.ones((predicted_steps, 1)), y_pred))
        u_in = np.ones((2,1))
        z = np.zeros((1 + self.k_dimensions + self.resevoir_size, 1))

        # Use last output and and last activation as new input and activation
        if not self.trained:
            self.train()
        u_in[1] = self.data[self.training_time]
        x = self.x_prev

        for i in tqdm(range(predicted_steps), disable=self.disable_tqdm):

            x_leak = (1 - self.leaking_rate) * x
            x = x_leak + self.leaking_rate * np.tanh(self.W_in @ u_in + self.R @ x)

            z[:self.k_dimensions + 1] = u_in
            z[self.k_dimensions + 1:] = x

            u_in[1] = self.W_out.T @ z
            y_pred[i] = u_in[1]

        n = predicted_steps
        y_pred = y_pred[:,1]
        err = np.abs(self.data[self.training_time:self.training_time+n] - y_pred[:n][:,None])

        rse = np.sqrt(err.T @ err)[0,0]
        
        if plot:
            self.__plot_predictions(predicted_steps, y_pred, err)
        if return_predicted:
            return y_pred, rse
        else:
            return rse


    def __plot_predictions(self, *args) -> None:
        predicted_steps, y_pred, err = args

        ''' Plotting the prediction '''

        n = predicted_steps

        plt.figure(figsize=(15,10))
        plt.subplot(2,1,1)
        plt.plot(self.data[self.training_time:self.training_time+n])
        plt.plot(y_pred[:n], alpha=0.9);

        plt.subplot(2,1,2)
        plt.plot(err, alpha=0.9, color='firebrick')

        rse = np.sqrt(err.T @ err)[0,0]
        plt.title('rse = {0:.4f}'.format(rse))