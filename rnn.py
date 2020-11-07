from abc import ABC, abstractmethod
import numpy as np
from utils import *


class RNN(ABC):
    """
    Base RNN class
    """
    _timestep = 1
    _named_parameters = None
    _initializer = None
    _hidden_states = []
    _cell_states = []
    _outputs = []
    
    def __init__(self, a_init=None, c_init=None):
        self.initialize_parameters(a_init, c_init)
           
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
    
    def initialize_parameters(self, a_init, c_init):
        """
            Initialize hidden state and cell parameters
            Initialize initial hidden and cell states
        """
        initializer_fn = getattr(self, self._initializer)
        self._named_parameters = initializer_fn()
        if a_init is not None:
            self._hidden_states.append(a_init)
        
        if c_init is not None:
            self._cell_states.append(c_init)


class VanillaRNN(RNN):
    """
    Vanilla RNN class
    """
    def __init__(self, input_dims,
                       output_dims,
                       a_init=None, 
                       initializer="random_initializer"):
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._hidden_state_dims = a_init.shape[0]
        self._initializer = initializer
        super(VanillaRNN, self).__init__(a_init)
        
    
    def __call__(self, X):
        a_prev = self.prev_hidden_state
        x = X[:, :, self._timestep - 1]
        return self.forward(x, a_prev)
        
    
    @property
    def cell_state(self):
        raise NotImplementedError("Vanilla RNNs have no cell state")
    

    def forward(self, x, a_prev):
        weights = np.hstack([self.named_parameters["Waa"], self.named_parameters["Wax"]])
        inputs = np.vstack([a_prev , x])
        Wy = self.named_parameters["Wya"]
        ba = self.named_parameters["ba"]
        by = self.named_parameters["by"]
        a_next = tanh(weights.dot(inputs) + ba)
        y = softmax(Wy.dot(a_next) + by)
        self._hidden_states.append(a_next)
        self._outputs.append(y)
        self._timestep += 1
        return a_next, y
    
    def backward(self):
        pass
    
    def random_initializer(self):
        named_parameters = {
            "Wax": np.random.rand(self._hidden_state_dims, self._input_dims),
            "Waa": np.random.rand(self._hidden_state_dims, self._hidden_state_dims),
            "Wya": np.random.rand(self._output_dims, self._hidden_state_dims),
            "ba" : np.random.rand(self._hidden_state_dims, 1),
            "by" : np.random.rand(self._output_dims, 1)
        }
        return named_parameters



class LSTM(RNN):
    """
    LSTM RNN class

    """
    def __init__(self, input_dims,
                       output_dims,
                       a_init=None,
                       c_init=None, 
                       initializer="random_initializer"):
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._hidden_state_dims = a_init.shape[0]
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
            "Wf": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._input_dims),
            "Wi": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._input_dims),
            "Wc": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._input_dims),
            "Wo": np.random.rand(self._hidden_state_dims, self._hidden_state_dims + self._input_dims),
            "Wy": np.random.rand(self._output_dims, self._hidden_state_dims),
            "bf" : np.random.rand(self._hidden_state_dims, 1),
            "bi" : np.random.rand(self._hidden_state_dims, 1),
            "bc" : np.random.rand(self._hidden_state_dims, 1),
            "bo" : np.random.rand(self._hidden_state_dims, 1),
            "by" : np.random.rand(self._output_dims, 1),
        }
        return named_parameters


class SequenceModel:
    """
    Sequence Model
    
    Args:
        num_time_steps - number of time steps
        rnn_model - rnn model object
    """
    def __init__(self, num_time_steps, rnn_model):
        self.num_time_steps = num_time_steps
        self.model = rnn_model
            
    def __call__(self, X):
        for i in range(self.num_time_steps):
            outputs = self.model(X) 
        return outputs

    def get_timestep_outputs(self, timestep):
        output = {
            "hidden_state": self.model._hidden_states[timestep - 1],
            "y_output": self.model._outputs[timestep - 1]
        }
        
        if len(self.model._cell_states) != 0:
            output.update({"cell_state": self.model._cell_states[timestep - 1]})

        return output
