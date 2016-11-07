import numpy as np
from artemis.fileman.experiment_record import experiment_root
from artemis.general.should_be_builtins import izip_equal
from artemis.plotting.db_plotting import dbplot, dbplot_hang, set_dbplot_default_layout, \
    hold_dbplots
from artemis.plotting.matplotlib_backend import LinePlot, Moving2DPointPlot
from plato.tools.optimization.optimizers import GradientDescent
from sigma_delta.mlp.forward_pass import quantized_forward_pass_cost_and_output
from artemis.ml.tools.neuralnets import initialize_network_params
from sigma_delta.mlp.comp_error_scale_optimizer import CompErrorScaleOptimizer
from artemis.ml.tools.iteration import minibatch_iterate_info


"""
In this demo we create a random network and run the rescaling algorithm on it.

We compare the results for a range of error-computation tradeoffs to a large set of randomly drawn
scales to verify that our learned scale values fall on the error-computation pareto front.
"""


@experiment_root
def demo_converge_to_pareto_curve(
        layer_sizes = [100, 100, 100, 100],
        w_scales = [1, 1, 1],
        n_samples = 100,
        learning_rate = 0.01,
        n_epochs = 100,
        minibatch_size = 10,
        n_random_points_to_try = 1000,
        random_scale_range = (1, 5),
        parametrization = 'log',
        computation_weights = np.logspace(-6, -3, 8),
        layerwise_scales = True,
        show_random_scales = True,
        error_loss = 'L1',
        hang_now=True,
        seed=1234):

    set_dbplot_default_layout('h')

    rng = np.random.RandomState(seed)
    ws = initialize_network_params(layer_sizes=layer_sizes, mag='xavier-relu', include_biases=False, rng=rng)
    ws = [w*s for w, s in izip_equal(ws, w_scales)]
    train_data = rng.randn(n_samples, layer_sizes[0])
    _, true_out = quantized_forward_pass_cost_and_output(train_data, weights=ws, scales=None, quantization_method=None, seed = 1234)

    # Run the random search
    scales_to_try = np.abs(rng.normal(loc = np.mean(random_scale_range), scale=np.diff(random_scale_range), size=(n_random_points_to_try, len(ws))))

    if show_random_scales:
        ax=dbplot(scales_to_try.T, 'random_scales', axis='Scales', plot_type=lambda: LinePlot(plot_kwargs=dict(color=(.6, .6, .6)), make_legend=False), xlabel = 'Layer', ylabel='Scale')
    ax.set_xticks(np.arange(len(w_scales)))

    random_flop_counts, random_errors = compute_flop_errors_for_scales(train_data, scales_to_try, ws=ws, quantization_method='round', true_out=true_out, seed=1234)
    dbplot((random_flop_counts/1e3/len(train_data), random_errors), 'random_flop_errors', axis='Tradeoff', xlabel='kOps/sample', ylabel='Error',
           plot_type = lambda: LinePlot(plot_kwargs=dict(color=(.6, .6, .6), marker='.', linestyle=' ')))

    # Now run with optimization, across several values of K (total scale)
    for comp_weight in computation_weights:
        net = CompErrorScaleOptimizer(ws, optimizer=GradientDescent(learning_rate), comp_weight=comp_weight, layerwise_scales=layerwise_scales,
                hidden_activations='relu', output_activation='relu', parametrization=parametrization, rng=rng)
        f_train = net.train_scales.partial(error_loss=error_loss).compile()
        f_get_scales = net.get_scales.compile()
        for training_minibatch, iter_info in minibatch_iterate_info(train_data, minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=np.arange(0, n_epochs, 1)):
            if iter_info.test_now:
                ks = f_get_scales()
                with hold_dbplots():
                    if show_random_scales:
                        dbplot(ks, 'solution_scales '+str(comp_weight), axis='Scales', plot_type=lambda: LinePlot(plot_kwargs=dict(linewidth=3), make_legend=False, axes_update_mode='expand'))
                    current_flop_counts, current_outputs = quantized_forward_pass_cost_and_output(train_data, weights=ws, scales=ks, quantization_method='round', seed = 1234)
                    current_error = np.abs(current_outputs-true_out).mean() / np.abs(true_out).mean()
                    if np.isnan(current_error):
                        print 'ERROR IS NAN!!!'
                    dbplot((current_flop_counts/1e3/len(train_data), current_error), 'k=%.3g curve' % (comp_weight, ), axis='Tradeoff',
                           plot_type = lambda: Moving2DPointPlot(legend_entries='$\\lambda=%.3g$' % comp_weight, axes_update_mode='expand', legend_entry_size=11))
            f_train(training_minibatch)

    if hang_now:
        dbplot_hang()


def compute_flop_errors_for_scales(data, scales_to_try, ws, quantization_method, seed, true_out = None):
    if true_out is None:
        _, true_out = quantized_forward_pass_cost_and_output(data, weights=ws, scales=[1.]*len(ws), seed=None, quantization_method=None)
    results = []
    for ks in scales_to_try:
        n_flops, output = quantized_forward_pass_cost_and_output(data, weights=ws, scales=ks, quantization_method=quantization_method, seed = seed)
        error = np.abs(output-true_out).mean() / np.abs(true_out).mean()
        results.append((n_flops, error))
    flop_counts, errors = (np.array(z) for z in zip(*results))
    return flop_counts, errors


demo_converge_to_pareto_curve.add_variant('paper-corrected-figure-poorscaled',
    computation_weights = [10**i for i in np.arange(-7, -1)],
    parametrization = 'log',
    learning_rate=0.003,
    n_epochs=200,
    error_loss='L2',
    w_scales = [.5, 8, .25],
    )


if __name__ == '__main__':

    demo_converge_to_pareto_curve.get_variant('paper-corrected-figure-poorscaled').display_or_run()
