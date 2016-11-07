from collections import OrderedDict
import numpy as np
from artemis.fileman.experiment_record import ExperimentFunction
from artemis.general.numpy_helpers import get_rng
from artemis.plotting.db_plotting import dbplot, set_dbplot_figure_size, hold_dbplots
from artemis.plotting.matplotlib_backend import LinePlot
from plato.tools.optimization.optimizers import GradientDescent
from sigma_delta.mlp.forward_pass import quantized_forward_pass_cost_and_output
from artemis.ml.tools.neuralnets import forward_pass
from sigma_delta.mlp.pretrained_nets import train_conventional_mlp_on_mnist
from sigma_delta.mlp.comp_error_scale_optimizer import CompErrorScaleOptimizer
from sigma_delta.mlp.measure_mnist_results import get_mnist_results_with_parameters, display_discrete_network_results
from artemis.ml.predictors.train_and_test import percent_argmax_incorrect
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.tools.iteration import minibatch_iterate_info
__author__ = 'peter'


"""
In this experiment we start with a network that is pretrained on MNIST.  We then learn the scales for a range of
error-computation tradeoff values.  We compare the computation of both the Sigma-Delta and the Rounding Versions of the optimally-scaled network
to that of the original network, for both the MNIST and Temporal MNIST Dataset.
"""


@ExperimentFunction(display_function=lambda results: display_discrete_network_results(results), is_root=True)
def demo_optimize_mnist_net(
        hidden_sizes = [200, 200],
        learning_rate = 0.01,
        n_epochs = 100,
        minibatch_size = 10,
        parametrization = 'log',
        computation_weights = np.logspace(-6, -3, 8),
        layerwise_scales = True,
        show_scales = True,
        hidden_activations='relu',
        test_every = 0.5,
        output_activation='softmax',
        error_loss = 'L1',
        comp_evaluation_calc='multiplyadds',
        smoothing_steps=1000,
        seed=1234):

    train_data, train_targets, test_data, test_targets = get_mnist_dataset(flat=True).to_onehot().xyxy

    params = train_conventional_mlp_on_mnist(hidden_sizes=hidden_sizes, hidden_activations=hidden_activations, output_activation=output_activation, rng=seed)
    weights, biases = params[::2], params[1::2]

    rng = get_rng(seed+1)

    true_out = forward_pass(input_data = test_data, weights = weights, biases=biases, hidden_activations=hidden_activations, output_activation=output_activation)
    optimized_results = OrderedDict([])
    optimized_results['unoptimized'] = get_mnist_results_with_parameters(weights=weights, biases=biases, scales = None, hidden_activations=hidden_activations, output_activation=output_activation, smoothing_steps=smoothing_steps)

    set_dbplot_figure_size(15, 10)
    for comp_weight in computation_weights:
        net = CompErrorScaleOptimizer(ws=weights, bs=biases, optimizer=GradientDescent(learning_rate), comp_weight=comp_weight,
            layerwise_scales=layerwise_scales, hidden_activations=hidden_activations, output_activation=output_activation,
            parametrization=parametrization, rng=rng)
        f_train = net.train_scales.partial(error_loss=error_loss).compile()
        f_get_scales = net.get_scales.compile()
        for training_minibatch, iter_info in minibatch_iterate_info(train_data, minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=np.arange(0, n_epochs, test_every)):
            if iter_info.test_now:  # Test the computation and all that
                ks = f_get_scales()
                print 'Epoch %.3g' % (iter_info.epoch, )
                with hold_dbplots():
                    if show_scales:
                        if layerwise_scales:
                            dbplot(ks, '%s solution_scales' % (comp_weight, ), plot_type=lambda: LinePlot(plot_kwargs=dict(linewidth=3), make_legend=False, axes_update_mode='expand', y_bounds=(0, None)), axis='solution_scales', xlabel='layer', ylabel='scale')
                        else:
                            for i, k in enumerate(ks):
                                dbplot(k, '%s solution_scales' % (i, ), plot_type=lambda: LinePlot(plot_kwargs=dict(linewidth=3), make_legend=False, axes_update_mode='expand', y_bounds=(0, None)), axis='solution_scales', xlabel='layer', ylabel='scale')
                    current_flop_counts, current_outputs = quantized_forward_pass_cost_and_output(test_data, weights=weights,
                        scales=ks, quantization_method='round', hidden_activations=hidden_activations, output_activation=output_activation,
                        computation_calc=comp_evaluation_calc, seed = 1234)
                    current_error = np.abs(current_outputs-true_out).mean() / np.abs(true_out).mean()
                    current_class_error = percent_argmax_incorrect(current_outputs, test_targets)
                    if np.isnan(current_error):
                        print 'ERROR IS NAN!!!'
                    dbplot((current_flop_counts/1e6, current_error), '%s error-curve' % (comp_weight, ), axis= 'error-curve', plot_type='trajectory+', xlabel='MFlops', ylabel='error')
                    dbplot((current_flop_counts/1e6, current_class_error), '%s class-curve' % (comp_weight, ), axis= 'class-curve', plot_type='trajectory+', xlabel='MFlops', ylabel='class-error')
            f_train(training_minibatch)
        optimized_results['lambda=%.3g' % (comp_weight, )] = get_mnist_results_with_parameters(weights=weights, biases=biases, scales = ks, hidden_activations=hidden_activations, output_activation=output_activation, smoothing_steps=smoothing_steps)
    return optimized_results


demo_optimize_mnist_net.add_variant('paper_version', learning_rate=0.001, layerwise_scales = True, computation_weights=np.logspace(-10, -5, 10), n_epochs=30, parametrization='log', error_loss='KL')

if __name__ == '__main__':
    demo_optimize_mnist_net.get_variant('paper_version').display_or_run()
