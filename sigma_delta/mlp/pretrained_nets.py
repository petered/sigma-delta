from artemis.fileman.disk_memoize import memoize_to_disk
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.cost import get_named_cost_function
from plato.tools.optimization.optimizers import get_named_optimizer
from artemis.ml.tools.neuralnets import initialize_network_params
from artemis.ml.predictors.predictor_comparison import assess_online_predictor
from artemis.ml.datasets.mnist import get_mnist_dataset

__author__ = 'peter'

@memoize_to_disk
def train_conventional_mlp_on_mnist(hidden_sizes, n_epochs=50, w_init='xavier-both', minibatch_size=20, rng=1234, optimizer = 'sgd',
        hidden_activations = 'relu', output_activation = 'softmax', learning_rate = 0.01, cost_function = 'nll', use_bias = True,
        l1_loss=0, l2_loss=0, test_on = 'training+test'):

    dataset = get_mnist_dataset(flat=True)\

    if output_activation != 'softmax':
        dataset=dataset.to_onehot()

    all_layer_sizes = [dataset.input_size]+hidden_sizes+[dataset.n_categories]
    weights = initialize_network_params(layer_sizes=all_layer_sizes, mag=w_init, base_dist='normal', include_biases=False, rng=rng)
    net = MultiLayerPerceptron(
        weights = weights,
        hidden_activation=hidden_activations,
        output_activation=output_activation,
        use_bias=use_bias
        )
    predictor = GradientBasedPredictor(
        function = net,
        cost_function=get_named_cost_function(cost_function),
        optimizer=get_named_optimizer(optimizer, learning_rate=learning_rate),
        regularization_cost=lambda params: sum(l1_loss*abs(p_).sum() + l2_loss*(p_**2).sum() if p_.ndim==2 else 0 for p_ in params)
        ).compile()
    assess_online_predictor(predictor=predictor, dataset=dataset, evaluation_function='percent_argmax_correct', test_epochs=range(0, n_epochs, 1), test_on=test_on, minibatch_size=minibatch_size)
    ws = [p.get_value() for p in net.parameters]
    return ws


def train_conventional_relu_mlp_on_mnist(hidden_sizes, output_activation='relu', cost_function = 'mse', **kwargs):
    return train_conventional_mlp_on_mnist(hidden_sizes=hidden_sizes, use_bias=False, hidden_activations='relu',
        cost_function=cost_function, output_activation=output_activation, **kwargs)



    #
    # dataset = get_mnist_dataset(flat=True).to_onehot()
    # net = MultiLayerPerceptron.from_init(
    #     w_init=w_init,
    #     rng=rng,
    #     layer_sizes=[dataset.input_size]+hidden_sizes+[dataset.target_size],
    #     hidden_activation = 'relu',
    #     output_activation = output_activation,
    #     use_bias=False
    #     )
    # predictor = GradientBasedPredictor(
    #     function = net,
    #     cost_function=mean_squared_error,
    #     optimizer=get_named_optimizer(optimizer, learning_rate=learning_rate)
    #     ).compile()
    # assess_online_predictor(predictor=predictor, dataset=dataset, evaluation_function='percent_argmax_correct', test_epochs=range(0, n_epochs, 1), test_on='test', minibatch_size=minibatch_size)
    # ws = [p.get_value() for p in net.parameters]
    # return ws