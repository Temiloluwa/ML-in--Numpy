from abc import ABC, abstractmethod
import numpy as np
from utils import *


class RNN(ABC):
    """
    Base RNN class
    """
    _timestep = 1
    _named_parameters = None
    _parameters_initialized = False
    _initializer = None
    _hidden_state_dims = None
    _hidden_states = []
    _outputs = []
    _cell_states = []
    
    def __init__(self, is_lstm=False, stateful=False):
        self._is_lstm = is_lstm
        self._stateful = stateful
        
           
    @property
    def hidden_state(self):
        return self._hidden_states[self._timestep]
    
    @property
    def prev_hidden_state(self):
        return self._hidden_states[self._timestep - 1]
    
    @property
    def cell_state(self):
        return self._cell_states[self._timestep]

    @property
    def prev_cell_state(self):
        return self._cell_states[self._timestep - 1]
    
    @property
    def output(self):
        return self._outputs[self._timestep]
    
    @property
    def named_parameters(self):
        return self._named_parameters

       
    @abstractmethod
    def random_initializer(self):
        pass
    
    @abstractmethod
    def forward(self, x, a_prev, c_prev=None):
        pass
    
    @abstractmethod
    def backward(self):
        pass


    def __call__(self, X):
        if X.ndim < 3:
            raise ValueError("Inputs should have at least 3 dimensions")
        elif X.ndim > 3:
            new_shape = (-1,) + X.shape[-2:]
            X = X.reshape(new_shape)

        if not self._parameters_initialized:
            self.initialize_parameters()
            self._parameters_initialized = True
        
        bs, sequence_len, _ = X.shape
        a_prev = np.zeros((bs, self._hidden_state_dims))
        c_prev = np.zeros((bs, self._hidden_state_dims)) if self._is_lstm else None
        self.reset_states(a_prev, c_prev)

        for _ in range(sequence_len):
            x = X[:, self._timestep - 1, :]
            if self._is_lstm:
                a_prev, c_prev, y = self.forward(x, a_prev, c_prev)
                self._cell_states.append(c_prev)
            else:
                a_prev, y = self.forward(x, a_prev)

            self._hidden_states.append(a_prev)
            self._outputs.append(y)
            self._timestep += 1
        self._timestep = 0
        
    
    def initialize_parameters(self):
        """
            Initialize hidden state and cell parameters
            Initialize initial hidden and cell states
        """
        initializer_fn = getattr(self, self._initializer)
        self._named_parameters = initializer_fn()


    def reset_states(self, a_prev, c_prev):
        if self._stateful and self._hidden_states is not []:
            self._hidden_states = [self._hidden_states[0]]
            if self._is_lstm:
                self._cell_states = [self._cell_states[0]]
        else:
            self._hidden_states = [a_prev]
            if self._is_lstm:
                self._cell_states = [c_prev]        


class VanillaRNN(RNN):
    """
    Vanilla RNN class
    """
    def __init__(self, hidden_state_dims,
                       output_dims,
                       initializer="random_initializer",
                       is_lstm=False, 
                       stateful=False):
        self._hidden_state_dims = hidden_state_dims
        self._output_dims = output_dims
        self._initializer = initializer
        super(VanillaRNN, self).__init__(is_lstm, stateful)
        
    
    @property
    def cell_state(self):
        raise NotImplementedError("Vanilla RNNs have no cell state")

    def __call__(self, 
                X, 
                return_sequences=False):

        self._embed_dims = X.shape[-1]
        super(VanillaRNN, self).__call__(X)
        return self._hidden_states[-1], self._outputs[-1]
       
        
    def forward(self, x, a_prev):
        weights = np.vstack([self.named_parameters["Waa"], self.named_parameters["Wxa"]])
        inputs = np.hstack([a_prev , x])
        Wy = self.named_parameters["Way"]
        ba = self.named_parameters["ba"]
        by = self.named_parameters["by"]
        a_next = tanh(inputs.dot(weights) + ba)
        y = softmax(a_next.dot(Wy) + by)
        return a_next, y
        
    
    def backward(self):
        pass
    
    def random_initializer(self, kwargs):
        named_parameters = {
            "Wxa": np.random.rand(self._embed_dims, self._hidden_state_dims),
            "Waa": np.random.rand(self._hidden_state_dims, self._hidden_state_dims),
            "Way": np.random.rand(self._hidden_state_dims, self._output_dims),
            "ba" : np.random.rand(1, self._hidden_state_dims),
            "by" : np.random.rand(1, self._output_dims)
        }

        return named_parameters

    def ones_initializer(self):
        named_parameters = {
            "Wxa": np.ones((self._embed_dims, self._hidden_state_dims)),
            "Waa": np.ones((self._hidden_state_dims, self._hidden_state_dims)),
            "Way": np.ones((self._hidden_state_dims, self._output_dims)),
            "ba" : np.ones((1, self._hidden_state_dims)),
            "by" : np.ones((1, self._output_dims))
        }

        return named_parameters




class LSTM(RNN):
    """
    LSTM RNN class

    """
    def __init__(self, embed_dims,
                       output_dims,
                       a_init=None,
                       c_init=None, 
                       initializer="random_initializer"):
        self._embed_dims = embed_dims
        self._output_dims = output_dims
        self._initializer = initializer
        super(LSTM, self).__init__(a_init, c_init)
    
    
    def __call__(self, X):
        a_prev = self.prev_hidden_state
        c_prev = self.prev_cell_state
        x = X[:, :, self._timestep - 1]
        return self.forward(x, a_prev, c_prev)

        
    def forward(self, x, a_prev, c_prev):
        concat = np.concatenate([a_prev, x], axis=0)
        
        Wf = self.named_parameters["Wf"]
        bf = self.named_parameters["bf"]
        Wi = self.named_parameters["Wi"]
        bi = self.named_parameters["bi"]
        Wc = self.named_parameters["Wc"]
        bc = self.named_parameters["bc"]
        Wo = self.named_parameters["Wo"]
        bo = self.named_parameters["bo"]
        Wy = self.named_parameters["Wy"]
        by = self.named_parameters["by"]
        
        forget_gate = sigmoid(Wf.dot(concat) + bf)
        update_gate = sigmoid(Wi.dot(concat) + bi)
        output_gate = sigmoid(Wo.dot(concat) + bo)
        candidate = tanh(Wc.dot(concat) + bc)
        c_next = forget_gate*c_prev + update_gate*candidate
        a_next = output_gate * tanh(c_next)
        y = softmax(Wy.dot(c_next) + by)
        self._cell_states.append(c_next)
        self._hidden_states.append(a_next)
        self._outputs.append(y)
        self._timestep += 1
        return a_next, c_next, y
    
    def backward(self):
        pass
    
    def random_initializer(self):
        named_parameters = {
            "Wf": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._embed_dims),
            "Wi": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._embed_dims),
            "Wc": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._embed_dims),
            "Wo": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._embed_dims),
            "Wy": np.random.rand(self._output_dims, self._hidden_state_dims),
            "bf" : np.random.rand(self._hidden_state_dims, 1),
            "bi" : np.random.rand(self._hidden_state_dims, 1),
            "bc" : np.random.rand(self._hidden_state_dims, 1),
            "bo" : np.random.rand(self._hidden_state_dims, 1),
            "by" : np.random.rand(self._output_dims, 1),
        }
        return named_parameters

