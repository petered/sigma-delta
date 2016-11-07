from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as tt
from artemis.general.should_be_builtins import izip_equal
from plato.core import create_shared_variable, symbolic, add_update
from plato.interfaces.helpers import get_theano_rng
from plato.tools.convnet.conv_specifiers import ConvolverSpec
from plato.tools.convnet.convnet import specifier_to_layer, ConvLayer

__author__ = 'peter'


class DiscretizedForwardPass(object):

    def __init__(self, named_layers):
        self.named_layers = named_layers
        self._func = self._get_cost_and_output.compile()

    def get_cost_and_output(self, x, scales):
        for scale_value, scaled_rounder in izip_equal(scales, [layer for layer in self.named_layers.values() if isinstance(layer, (ScaledRoundingLayer, ScaledHerdingLayer))]):
            scaled_rounder.set_scale(scale_value)

        comp_costs, outputs = zip(*[self._func(xi[None]) for xi in x])  # Can't just self._func(x) for memory reasons
        return np.concatenate(comp_costs), np.concatenate(outputs)

        # return self._func(x)

    @symbolic
    def _get_cost_and_output(self, x):
        """
        Do a forward pass of the network and compute the cost and output for a given input frame
        :return:
        """
        total_ops = 0
        for (name, layer), next_layer in zip(self.named_layers.iteritems(), self.named_layers.values()[1:]+[None]):
            if isinstance(layer, (ScaledRoundingLayer, ScaledHerdingLayer)):  # Our layer is a rounding layer.
                assert isinstance(next_layer, ConvLayer)
                signals = layer.get_all_signals(x)  # Well look at the fancy rounding layer
                total_ops += abs(signals['spikes']).flatten(2).sum(axis=1) * get_conv_layer_fanout(next_layer.w.shape, conv_mode={0:'full', 1:'same'}[next_layer.border_mode])
                # NOTE: The Border-mode thing above is NOT necessarily correct... it just happens to be true for VGGnet that all "valid" convolutions are to 1x1 maps.
                x = signals['output']
            else:
                x = layer(x)
        return total_ops, x


class RoundConvNetForwardPass(DiscretizedForwardPass):

    def __init__(self, conv_specs):
        DiscretizedForwardPass.__init__(self, conv_specs_to_round_layers(conv_specs))


class SigmaDeltaConvNetForwardPass(DiscretizedForwardPass):

    def __init__(self, conv_specs, input_shape):
        DiscretizedForwardPass.__init__(self, conv_specs_to_td_specs(conv_specs, input_shape=input_shape))

    def get_cost_and_output(self, x, scales, reset_state = True):

        if reset_state:
            for v in self.named_layers.values():
                if isinstance(v, (ScaledHerdingLayer, TemporalIntegrator, TemporalDifference, OnceAddConvLayer)):
                    v.reset()

        for scale_value, scaled_rounder in izip_equal(scales, [layer for layer in self.named_layers.values() if isinstance(layer, (ScaledRoundingLayer, ScaledHerdingLayer))]):
            scaled_rounder.set_scale(scale_value)
        # First: Reset states and OnceAddConvLayers
        comp_costs, outputs = zip(*[self._func(xi[None]) for xi in x])
        return np.concatenate(comp_costs), np.concatenate(outputs)
        # for scale_value, scaled_rounder in izip_equal(scales, [layer for layer in self.named_layers.values() if isinstance(layer, ScaledRoundingLayer)]):
        #     scaled_rounder.set_scale(scale_value)
        # return self._func(x)


def get_conv_layer_fanout(weight_shape, conv_mode):
    """
    The the "fan-out" of an input feeding into
    :param weight_shape: A 4-element tuple, identifiying (n_output_maps, n_input_maps, size_y, size_x)
    :param conv_mode: Can be:
        'full': Meaning that the output is (n_samples, n_output_maps, 1, 1)
        'same': Meaning that the output is (n_samples, n_output_maps, input_size_y, input_size_x)
    :return:
    """
    n_output_maps, n_input_maps, size_y, size_x = weight_shape
    assert conv_mode in ('full', 'same')
    if conv_mode=='full':
        return n_output_maps
    elif conv_mode == 'same':
        return n_output_maps*size_y*size_x



def conv_specs_to_round_layers(convnet_specifiers):
    if isinstance(convnet_specifiers, (list, tuple)):
        convnet_specifiers = OrderedDict(enumerate(convnet_specifiers))
    layers = OrderedDict()
    # layers['post-input-disc'] = ScaledRoundingLayer()
    last_name = 'input'
    for name, spec in convnet_specifiers.iteritems():
        convolver_up_next = isinstance(spec, ConvolverSpec)
        if convolver_up_next:
            layers['post-'+last_name+'-disc'] = ScaledRoundingLayer()
        layers[name] = specifier_to_layer(spec)
        last_name = name
    return layers


def conv_specs_to_td_specs(convnet_specifiers, input_shape, target_shape=None):

    assert isinstance(input_shape, (list, tuple)) and len(input_shape)==3, 'Input shape must specify (n_maps, size_y, size_x).  We got %s' % (input_shape, )
    if target_shape is not None:  # Why do we even take this argument?
        assert isinstance(target_shape, (list, tuple)) and len(target_shape)==3, 'Target shape must specify (n_maps, size_y, size_x).  We got %s' % (target_shape, )
    if isinstance(convnet_specifiers, (list, tuple)):
        convnet_specifiers = OrderedDict(enumerate(convnet_specifiers))
    in_shape = (1, )+input_shape
    layers = OrderedDict()
    layers['post-input-diff'] = TemporalDifference((1, )+input_shape)
    layers['post-input-disc'] = ScaledHerdingLayer(in_shape)
    in_intdiff_block = False
    last_name = None
    for name, spec in convnet_specifiers.iteritems():
        out_shape = spec.shape_transfer(in_shape)
        now_in_intdiff_block = not isinstance(spec, ConvolverSpec)
        if now_in_intdiff_block and not in_intdiff_block:
            layers['post-'+last_name+'-integrator'] = TemporalIntegrator(shape = in_shape)
        elif not now_in_intdiff_block and in_intdiff_block:
            layers['post-'+last_name+'-diff'] = TemporalDifference(shape = in_shape)
            layers['post-'+last_name+'-disc'] = ScaledHerdingLayer(in_shape)
        in_intdiff_block = now_in_intdiff_block
        if isinstance(spec, ConvolverSpec):  # It's overkill to wrap a ConvLayer.
            layers[name] = OnceAddConvLayer(
                w=spec.w,
                b=spec.b,
                border_mode= {'full': 0, 'same': 1, 'valid': 0}[spec.mode] if spec.mode in ('full', 'same', 'valid') else spec.mode,
                filter_flip=False
                )
        else:
            layers[name] = specifier_to_layer(spec)
        in_shape = out_shape
        last_name = name

    if not in_intdiff_block:
        layers['int_out'] = TemporalIntegrator(out_shape)
    return layers


def get_full_convnet_computational_cost(layer_specs, input_shape):
    """
    Get the total number of opts required to execute a full convnet.
    :return:
    """
    total_cost = 0
    in_shape = (1, )+input_shape
    for name, spec in layer_specs.iteritems():
        out_shape = spec.shape_transfer(in_shape)
        if isinstance(spec, ConvolverSpec):
            n_out, n_in, filter_size_y, filter_size_x = spec.w.shape
            fan_in = n_in*filter_size_x*filter_size_y
            total_cost += np.prod(out_shape)*fan_in * 2
            # For each unit, that's (fan_in multiplications, fan_in-1 additions, and 1 bias addition)
        in_shape = out_shape
        assert out_shape[2]>=1 and out_shape[3]>=1
    return total_cost


class ScaledRoundingLayer(object):

    def __init__(self, scale_shape = None):

        self.log_scales = create_shared_variable(0. if scale_shape is None else np.zeros(scale_shape))

    def get_scale(self):
        return tt.exp(self.log_scales)

    def set_scale(self, val):
        self.log_scales.set_value(np.log(val).astype(theano.config.floatX))

    def __call__(self, input_):
        return self.get_all_signals(input_)['output']

    def get_all_signals(self, input_, corruption_type = 'round', rng = None):
        scale = self.get_scale()
        scaled_input = input_*scale
        if corruption_type == 'round':
            epsilon = tt.round(scaled_input) - scaled_input
        elif corruption_type == 'randround':
            rng = get_theano_rng(rng)
            epsilon = tt.where(rng.uniform(scaled_input.shape)>(scaled_input % 1), tt.floor(scaled_input), tt.ceil(scaled_input))-scaled_input
            print 'STOCH ROUNDING'
        elif corruption_type == 'rand':
            rng = get_theano_rng(1234)
            epsilon = rng.uniform(scaled_input.shape)-.5
        else:
            raise Exception('fdsfsd')
        spikes = scaled_input + epsilon
        output = spikes / scale
        signals = dict(
            input=input_,
            scaled_input=scaled_input,
            spikes=spikes,
            epsilon=epsilon,
            output=output,
            )
        return signals

    # def get_all_signals(self, input_):
    #     scale = self.get_scale()
    #
    #
    #
    #     scaled_input = input_*scale
    #
    #
    #
    #     # epsilon = tt.round(scaled_input) - scaled_input
    #
    #     rng = get_theano_rng(1234)
    #     epsilon = rng.uniform(scaled_input.shape)-.5
    #
    #     spikes = scaled_input + epsilon
    #     output = spikes / scale
    #     signals = dict(
    #         input=input_,
    #         scaled_input=scaled_input,
    #         spikes=spikes,
    #         epsilon=epsilon,
    #         output=output,
    #         )
    #     return signals

    def get_scale_param(self):
        return self.log_scales


class ScaledHerdingLayer(object):

    def __init__(self, shape, scale_shape = None):
        self.phi = create_shared_variable(np.zeros(shape))
        self.log_scales = create_shared_variable(0. if scale_shape is None else np.zeros(scale_shape))

    def get_scale(self):
        return tt.exp(self.log_scales)

    def set_scale(self, val):
        self.log_scales.set_value(np.log(val))

    def __call__(self, input_):
        return self.get_all_signals(input_)['output']

    def get_all_signals(self, input_):
        scale = self.get_scale()
        scaled_input = input_*scale

        inc_phi = self.phi + scaled_input
        epsilon = tt.round(inc_phi) - inc_phi
        spikes = inc_phi + epsilon
        # spikes = tt.round(inc_phi)
        new_phi = inc_phi-spikes

        output = spikes / scale
        signals = dict(
            input=input_,
            scaled_input=scaled_input,
            spikes=spikes,
            epsilon=epsilon,
            output=output,
            )
        add_update(self.phi, new_phi)
        return signals

    def get_scale_param(self):
        return self.log_scales

    def reset(self):
        self.phi.set_value(np.zeros_like(self.phi.get_value()))


class SignedSpikingLayer(object):

    def __init__(self, shape, scale = 1):
        self.phi = create_shared_variable(np.zeros(shape))
        self.scale = scale

    def __call__(self, inputs):
        if self.scale != 1:
            import theano
            inputs = inputs * np.array(self.scale, dtype=theano.config.floatX)
        inc_phi = self.phi + inputs
        spikes = tt.round(inc_phi)
        new_phi = inc_phi-spikes
        add_update(self.phi, new_phi)
        return spikes


@symbolic
class OnceAddConvLayer(ConvLayer):

    def __init__(self, *args, **kwargs):
        ConvLayer.__init__(self, *args, **kwargs)
        self.bias_switch = create_shared_variable(1.)

    def __call__(self, x):
        """
        param x: A (n_samples, n_input_maps, size_y, size_x) image/feature tensor
        return: A (n_samples, n_output_maps, size_y-w_size_y+1, size_x-w_size_x+1) tensor
        """
        result = tt.nnet.conv2d(input=x, filters=self.w, border_mode=self.border_mode, filter_flip=self.filter_flip) + self.bias_switch*(self.b[:, None, None] if self.b is not False else 0)
        if self.b is not False:
            add_update(self.bias_switch, 0)
        return result

    def reset(self):
        self.bias_switch.set_value(1)


class IdentityLayer(object):

    def __call__(self, x):
        return x


class ScalingLayer(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        if self.scale == 1:
            return x
        else:
            return x*tt.constant(self.scale, dtype=theano.config.floatX)


class TemporalDifference(object):

    def __init__(self, shape):
        self.old = create_shared_variable(np.zeros(shape))

    def __call__(self, data):
        diff = data - self.old
        add_update(self.old, data)
        return diff

    def reset(self):
        self.old.set_value(np.zeros_like(self.old.get_value()))


class TemporalIntegrator(object):

    def __init__(self, shape):
        self.sum = create_shared_variable(np.zeros(shape))

    def __call__(self, data):
        new_sum = self.sum+data
        add_update(self.sum, new_sum)
        return new_sum

    def reset(self):
        self.sum.set_value(np.zeros_like(self.sum.get_value()))
