import numpy as np
import theano
from theano.gradient import disconnected_grad
import theano.tensor as tt
from artemis.general.should_be_builtins import izip_equal, bad_value
from plato.core import create_shared_variable, symbolic, tdbprint
from plato.interfaces.helpers import get_theano_rng, get_named_activation_function
from plato.tools.optimization.optimizers import get_named_optimizer

__author__ = 'peter'


"""
We construct a slighly artificial situation ... but it is relavent to our video recognition task.

We want to:
- Rescale the units in the network such that, GIVEN a random initial phi, the error is minimal, such that scaling terms
  add up to a fixed constant.
- If we are correct, this should correspond to the optimal

"""


class CompErrorScaleOptimizer(object):

    def __init__(self, ws, bs = None, comp_weight=1e-6, optimizer = None, layerwise_scales=False, parametrization = 'log',
                hidden_activations = 'relu', output_activation = 'softmax', rng = None):
        """
        Learns how to rescale the units to be an optimal rounding network.
        :param ws: A list of (n_in, n_out) weight matrices
        :param bs: A length of bias vectors (same length as ws)
        :param comp_weight: The weight (lambda in the paper) given to computation
        :param optimizer: The optimizer (an IGradientOptimizer object)
        :param layerwise_scales: Make scales layerwise (as opposed to unitwise)
        :param parametrization: What space to parametrize in ('log', 'direct', or 'softplus')
        :param hidden_activations: Hidden activation functions (as a string, eg 'relu')
        :param output_activation: Output activation function
        :param rng: Random number generator or seed.
        """
        if optimizer is None:
            optimizer = get_named_optimizer('sgd', 0.01)
        if bs is None:
            bs = [np.zeros(w.shape[1]) for w in ws]
        self.ws = [create_shared_variable(w) for w in ws]
        self.bs = [create_shared_variable(b) for b in bs]
        self.comp_weight = tt.constant(comp_weight, dtype=theano.config.floatX)
        self.optimizer = optimizer
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        scale_dims = [()]*len(ws) if layerwise_scales else [ws[0].shape[0]]+[w.shape[1] for w in ws[:-1]]
        self.k_params = \
            [create_shared_variable(np.ones(d)) for d in scale_dims] if parametrization=='direct' else \
            [create_shared_variable(np.zeros(d)) for d in scale_dims] if parametrization=='log' else \
            [create_shared_variable(np.zeros(d)+np.exp(1)-1) for d in scale_dims] if parametrization=='softplus' else \
            bad_value(parametrization)
        self.parametrization = parametrization
        self.rng = get_theano_rng(rng)

    @symbolic
    def get_scales(self):
        return \
            self.k_params if self.parametrization == 'direct' else \
            [tt.exp(k) for k in self.k_params] if self.parametrization == 'log' else \
            [tt.log(1+tt.exp(k)) for k in self.k_params] if self.parametrization == 'softplus' else \
            bad_value(self.parametrization)

    def compute_activations(self, input_data, do_round = True):
        layer_input = input_data
        layer_signals = []
        for i, (w, b, k) in enumerate(zip(self.ws, self.bs, self.get_scales())):
            scaled_input = layer_input*k
            if not do_round:
                eta=None
                spikes = scaled_input
            else:
                eta = tt.round(scaled_input) - scaled_input
                spikes = scaled_input + disconnected_grad(eta)
            nonlinearity = get_named_activation_function(self.hidden_activations if i<len(self.ws)-1 else self.output_activation)
            output = nonlinearity((spikes/k).dot(w)+b)
            layer_signals.append({'input': layer_input, 'scaled_input': scaled_input, 'eta': eta, 'spikes': spikes, 'output': output})
            layer_input = output
        return layer_signals

    @symbolic
    def predict(self, input_data, do_round=True):
        return self.compute_activations(input_data, do_round=do_round)[-1]['output']

    @symbolic
    def train_scales(self, input_data, error_loss = 'L1'):
        # Some tricky stuff in here.
        # When computing the error grad, we need to make all future units pass-through (otherwise grad will be zero)
        # When computing cost-grad, we:
        # - Only consider the local cost (nans seemed to happen when we considered total cost)
        # - Consider this layer to be pass-through (otherwise cost-grad will be zero)
        layer_signals = self.compute_activations(input_data, do_round=True)
        true_out = self.predict(input_data, do_round=False)
        error_loss = get_error_loss(guess=layer_signals[-1]['output'], truth=true_out, loss_type=error_loss)
        grads = []
        for i, (kp, k, sigs) in enumerate(izip_equal(self.k_params, self.get_scales(), layer_signals)):
            error_grad = tt.grad(error_loss, wrt=kp, consider_constant=[other_sigs['eta'] for other_sigs in layer_signals[i+1:]])  # PROBABLY NO GOOD
            layer_comp_loss = tt.switch(sigs['scaled_input']>0, sigs['spikes'], -sigs['spikes']).sum() * self.ws[i].shape[1]  # lets be safe
            comp_grad = tt.grad(layer_comp_loss, wrt=kp, consider_constant=[sigs['eta']])
            layer_grad = error_grad + self.comp_weight*comp_grad
            grads.append(layer_grad)
        self.optimizer.update_from_gradients(parameters=self.k_params, gradients=grads)


@symbolic
def get_error_loss(guess, truth, loss_type):
    if loss_type == 'L1':
        return abs(guess-truth).sum(axis=1).mean(axis=0)
    elif loss_type == 'L2':
        return ((guess-truth)**2).sum(axis=1).mean(axis=0)
    elif loss_type == 'KL':
        return (guess*(tt.log(guess)-tt.log(truth))).sum(axis=1).mean(axis=0)
    else:
        raise NotImplementedError(loss_type)
