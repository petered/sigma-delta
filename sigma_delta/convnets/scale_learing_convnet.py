from collections import OrderedDict

import theano.tensor as tt

from plato.core import symbolic, tdbprint
from plato.tools.convnet.convnet import ConvLayer
from sigma_delta.mlp.comp_error_scale_optimizer import get_error_loss
from sigma_delta.convnets.conv_forward_pass import \
    conv_specs_to_round_layers, ScaledRoundingLayer, get_conv_layer_fanout


__author__ = 'peter'


@symbolic
class ScaleLearningRoundingConvnet(object):

    def __init__(self, layers, optimizer, layerwise_scales=True):
        """
        layers is an OrdereDict of callables.
        """
        assert layerwise_scales, 'Only layerwise work now.'
        if isinstance(layers, (list, tuple)):
            layers = OrderedDict(enumerate(layers))
        else:
            assert isinstance(layers, OrderedDict), "Layers must be presented as a list, tuple, or OrderedDict"
        self.layers = layers
        self.optimizer = optimizer
        self.layerwise_scales = layerwise_scales

    def __call__(self, inp, do_round=True):
        return self.get_named_layer_activations(inp, do_round=do_round).values()[-1]

    def _compute_activations(self, x, do_round):
        rounding_signals = OrderedDict()
        named_activations = OrderedDict()
        for name, layer in self.layers.iteritems():
            if isinstance(layer, ScaledRoundingLayer):
                if not do_round:
                    continue
                else:
                    signals = layer.get_all_signals(x)  # Well look at the fancy rounding layer
                    rounding_signals[name] = signals
                    x = signals['output']
            else:
                x = layer(x)
            named_activations[name] = x
        return named_activations, rounding_signals

    @symbolic
    def train_scales(self, x, comp_weight, error_loss = 'L2'):
        true_activations, _ = self._compute_activations(x, do_round=False)
        approx_activations, rounding_signals = self._compute_activations(x, do_round=True)
        error_loss = get_error_loss(guess=approx_activations.values()[-1].flatten(2), truth=true_activations.values()[-1].flatten(2), loss_type=error_loss)
        param_grad_pairs = []
        for i, (layer_name, sigs) in enumerate(rounding_signals.iteritems()):
            assert isinstance(self.layers[layer_name], ScaledRoundingLayer), "You F'ed up."
            scale_param = self.layers[layer_name].get_scale_param()
            error_grad = tt.grad(error_loss, wrt=scale_param, consider_constant=[other_sigs['epsilon'] for other_sigs in rounding_signals.values()[i+1:]])
            next_layer = self.layers.values()[self.layers.values().index(self.layers[layer_name])+1]
            assert isinstance(next_layer, ConvLayer), "Again"
            layer_comp_loss = tt.switch(sigs['scaled_input']>0, sigs['spikes'], -sigs['spikes']).sum() \
                * get_conv_layer_fanout(next_layer.w.shape, conv_mode={0:'full', 1:'same'}[next_layer.border_mode])  # NOTE: NOT GENERAL: VGG SPECIFIC!
            comp_grad = tt.grad(layer_comp_loss, wrt=scale_param, consider_constant=[sigs['epsilon']])
            tdbprint(comp_weight*comp_grad, layer_name+'scaled comp grad')
            tdbprint(error_grad, layer_name+'scaled error grad')
            layer_grad = error_grad + comp_weight*comp_grad
            param_grad_pairs.append((scale_param, layer_grad))
        scale_params, grads = zip(*param_grad_pairs)
        self.optimizer.update_from_gradients(parameters=scale_params, gradients=grads)

    @symbolic
    def get_named_layer_activations(self, x, do_round=True):
        named_activations, _ = self._compute_activations(x, do_round=do_round)
        return named_activations

    @symbolic
    def get_scales(self):
        return [layer.get_scale() for layer in self.layers.values() if isinstance(layer, ScaledRoundingLayer)]

    @property
    def parameters(self):
        return [layer.get_scale_param() for layer in self.layers.values() if isinstance(layer, ScaledRoundingLayer)]

    def reset_scales(self):
        for p in self.parameters:
            p.set_value(0.)

    @staticmethod
    def from_convnet_specs(convnet_specifiers, optimizer):
        """
        Return a "ScaleLearningRoundingConvnet" convnet.

        :param convnet_specifiers: An OrderedDict<str, PrimitiveSpecifier>, identifying a convolutional network.
        :return: A SpikingDifferenceConvNet object
        """
        round_net_specs = conv_specs_to_round_layers(convnet_specifiers)
        net = ScaleLearningRoundingConvnet(round_net_specs, optimizer=optimizer, layerwise_scales=True)
        return net
