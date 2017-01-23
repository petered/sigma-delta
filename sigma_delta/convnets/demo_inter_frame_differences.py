from collections import OrderedDict
from artemis.experiments.experiment_record import ExperimentFunction
from artemis.experiments.ui import browse_experiments
import numpy as np
from artemis.general.ezprofile import EZProfiler
from artemis.general.mymath import cosine_distance
from artemis.plotting.db_plotting import dbplot, set_dbplot_figure_size

import matplotlib.pyplot as plt

__author__ = 'peter'


def _display_results(results):

    layerwise_differences = results['layerwise_differences']
    layerwise_norms = results['layerwise_norms']
    distance = results['distance']

    nonorms = np.all(np.isnan(layerwise_norms))

    plt.figure(figsize=(5, 7))
    plt.subplot(2,1,1)
    plt.xlabel('Frame #')
    plt.ylabel('Layer #')
    plt.imshow(layerwise_differences, interpolation='nearest', cmap='gray', aspect='auto')
    plt.title("Inter-Frame {} Distance".format(distance))
    plt.colorbar()

    legend_entries = ['Norms', 'Differences']

    # plt.subplot(3,1,2)
    # plt.plot(layerwise_norms.mean(axis=0), marker='.', linewidth=2, markersize=10)
    # plt.plot(layerwise_differences.mean(axis=0), marker='.', linewidth=2, markersize=10)
    # plt.xlabel('Frame #')
    # plt.ylabel('{} Distance'.format(distance))
    # if not nonorms:
    #     plt.legend(legend_entries, loc='best', framealpha=0.5)

    plt.subplot(2,1,2)
    plt.plot(layerwise_norms.mean(axis=1), marker='.', linewidth=2, markersize=10)
    plt.plot(layerwise_differences.mean(axis=1), marker='.', linewidth=2, markersize=10)
    plt.xlabel('Layer #')
    plt.ylabel('{} Distance'.format(distance))
    plt.xlim(0, layerwise_differences.shape[0]-1)
    if not nonorms:
        plt.legend(legend_entries, loc='best', framealpha=0.5)

    plt.show()


@ExperimentFunction(display_function=_display_results, is_root=True)
def demo_inter_frame_differences(post_nonlinearity = True, normalize=False, distance = 'cos'):
    """
    Run the scale optimization routine on a convnet.
    :param post_nonlinearity: Measure representations after the nonlinearity
    :return:
    """
    from plato.tools.pretrained_networks.vggnet import get_vgg_net
    from helpers.ilsvrc_data import get_vgg_video_splice

    test_videos, test_vgg_inputs = get_vgg_video_splice(['ILSVRC2015_train_00033009', 'ILSVRC2015_train_00033007'])

    set_dbplot_figure_size(12, 6)

    convnet = get_vgg_net()

    f = convnet.get_named_layer_activations.partial(include_input=True).compile()

    with EZProfiler('Computing activations'):
        activations = f(test_vgg_inputs)
    with EZProfiler('Computing Distances'):
        if post_nonlinearity:
            acts_of_interest = OrderedDict((k, a) for k, a in activations.iteritems() if k=='input' or k.startswith('relu') or k.startswith('prob'))
        else:
            acts_of_interest = OrderedDict((k, a) for k, a in activations.iteritems() if k=='input' or k.startswith('conv') or k.startswith('fc'))
        distance_function = {
            'cos': cosine_distance,
            'L2': lambda x, y, axis: np.sqrt(((x - y)**2).sum(axis=axis)),
            'L1': lambda x, y, axis: np.abs(x - y).sum(axis=axis),
            }[distance]
        n_frames = len(test_videos)
        if normalize:
            norm_function = {
                'cos': lambda x: x,
                'L2': lambda x: x/np.sqrt((x.reshape(x.shape[0], -1)**2).sum(axis=1).mean(axis=0)),
                'L1': lambda x: x/np.abs(x.reshape(x.shape[0], -1)).sum(axis=1).mean(axis=0),
                }[distance]
            acts_of_interest = OrderedDict((k, norm_function(v)) for k, v in acts_of_interest.iteritems())
        layerwise_differences = np.array([[distance_function(layer_frames[i], layer_frames[i+1], axis=None) for i in xrange(n_frames-1)] for layer_frames in acts_of_interest.values()])
        layerwise_norms = np.array([[distance_function(layer_frames[i], np.zeros_like(layer_frames[i]), axis=None) for i in xrange(1, n_frames)] for layer_frames in acts_of_interest.values()])

    return dict(layerwise_differences=layerwise_differences, layerwise_norms=layerwise_norms, distance=distance)

demo_inter_frame_differences.add_variant(distance='L1', normalize=True, post_nonlinearity=True)
demo_inter_frame_differences.add_variant(distance='cos', post_nonlinearity=True)


if __name__ == "__main__":
    browse_experiments()
    # demo_inter_frame_differences(distance='cos')
    # demo_inter_frame_differences(distance='L2', normalize=True, post_nonlinearity=True)

