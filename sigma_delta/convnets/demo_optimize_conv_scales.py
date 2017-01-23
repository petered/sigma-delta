from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import os
from artemis.fileman.experiment_record import experiment_function, experiment_root
from artemis.fileman.local_dir import get_local_path, make_dir
from artemis.general.ezprofile import EZProfiler
from artemis.general.mymath import cummean
from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.should_be_builtins import izip_equal
from artemis.ml.tools.iteration import minibatch_iterate_info
from artemis.plotting.db_plotting import dbplot, set_dbplot_figure_size, dbplot_hang
from artemis.plotting.matplotlib_backend import LinePlot
from helpers.ilsvrc_data import get_vgg_video_splice
from matplotlib.gridspec import GridSpec
from plato.tools.convnet.convnet import ConvNet
from plato.tools.optimization.optimizers import get_named_optimizer
from plato.tools.pretrained_networks.vggnet import get_vgg_layer_specifiers, get_vgg_label_at
from sigma_delta.convnets.conv_forward_pass import \
    RoundConvNetForwardPass, SigmaDeltaConvNetForwardPass, get_full_convnet_computational_cost
from sigma_delta.convnets.scale_learing_convnet import ScaleLearningRoundingConvnet

__author__ = 'peter'


@experiment_root
def demo_optimize_conv_scales(n_epochs=5, comp_weight = 1e-11, learning_rate=0.1, error_loss='KL', use_softmax=True, optimizer = 'sgd', shuffle_training=False):
    """
    Run the scale optimization routine on a convnet.  
    :param n_epochs:
    :param comp_weight:
    :param learning_rate:
    :param error_loss:
    :param use_softmax:
    :param optimizer:
    :param shuffle_training:
    :return:
    """
    if error_loss=='KL' and not use_softmax:
        raise Exception("It's very strange that you want to use a KL divergence on something other than a softmax error.  I assume you've made a mistake.")

    training_videos, training_vgg_inputs = get_vgg_video_splice(['ILSVRC2015_train_00033010', 'ILSVRC2015_train_00336001'], shuffle=shuffle_training, shuffling_rng=1234)
    test_videos, test_vgg_inputs = get_vgg_video_splice(['ILSVRC2015_train_00033009', 'ILSVRC2015_train_00033007'])

    set_dbplot_figure_size(12, 9)
    plt.subplots_adjust(left = 0.4)

    # Arrange plots
    gs1 = GridSpec(3, 1, top=1, bottom=0.45, hspace=0.06)
    gs2 = GridSpec(2, 1, top=0.35, bottom=.1, hspace=0.06)
    n_frames_to_show = 10
    display_frames = np.arange(len(test_videos)/n_frames_to_show/2, len(test_videos), len(test_videos)/n_frames_to_show)
    ax1=dbplot(np.concatenate(test_videos[display_frames], axis=1), "Test Videos", title = '', plot_type='pic', axis=gs1[0, 0])
    plt.subplots_adjust(wspace=0, hspace=.05)
    ax1.set_xticks(224*np.arange(len(display_frames)/2)*2+224/2)
    ax1.tick_params(labelbottom = 'on')

    layers = get_vgg_layer_specifiers(up_to_layer='prob' if use_softmax else 'fc8')

    # Setup the true VGGnet and get the outputs
    f_true = ConvNet.from_init(layers, input_shape = (3, 224, 224)).compile()
    true_test_out = flatten2(np.concatenate([f_true(frame_positions[None]) for frame_positions in test_vgg_inputs]))
    top5_true_guesses = argtopk(true_test_out, 5)
    true_guesses = np.argmax(true_test_out, axis=1)
    true_labels = [get_vgg_label_at(g, short=True) for g in true_guesses[display_frames[::2]]]
    full_convnet_cost = np.array([get_full_convnet_computational_cost(layer_specs=layers, input_shape=(3, 224, 224))]*len(test_videos))

    # Setup the approximate networks
    slrc_net = ScaleLearningRoundingConvnet.from_convnet_specs(layers, optimizer=get_named_optimizer(optimizer, learning_rate=learning_rate), corruption_type='rand', rng=1234)
    f_train_slrc = slrc_net.train_scales.partial(comp_weight=comp_weight, error_loss=error_loss).compile()
    f_get_scales = slrc_net.get_scales.compile()
    round_fp = RoundConvNetForwardPass(layers)
    sigmadelta_fp = SigmaDeltaConvNetForwardPass(layers, input_shape=(3, 224, 224))

    p = ProgressIndicator(n_epochs*len(training_videos))

    output_dir = make_dir(get_local_path('output/%T-convnet-spikes'))

    for input_minibatch, minibatch_info in minibatch_iterate_info(training_vgg_inputs, n_epochs=n_epochs, minibatch_size=1, test_epochs=np.arange(0, n_epochs, 0.1)):

        if minibatch_info.test_now:
            with EZProfiler('test'):
                current_scales = f_get_scales()
                round_costs, round_out = round_fp.get_cost_and_output(test_vgg_inputs, scales=current_scales)
                sd_costs, sd_out = sigmadelta_fp.get_cost_and_output(test_vgg_inputs, scales=current_scales)
                round_guesses, round_top1_correct, round_top5_correct = get_and_report_scores(round_costs.sum(axis=1), round_out, name='Round', true_top_1=true_guesses, true_top_k=top5_true_guesses)
                sd_guesses, sd_top1_correct, sd_top5_correct = get_and_report_scores(sd_costs.sum(axis=1), sd_out, name='SigmaDelta', true_top_1=true_guesses, true_top_k=top5_true_guesses)

                round_labels = [get_vgg_label_at(g, short=True) for g in round_guesses[display_frames[::2]]]

                ax1.set_xticklabels(['{}\n{}'.format(tg, rg) for tg, rg in izip_equal(true_labels, round_labels)])
                ax=dbplot(np.array([round_costs.sum(axis=1)/1e9, sd_costs.sum(axis=1)/1e9, full_convnet_cost.sum(axis=1)/1e9]).T, 'Computation',
                        plot_type='thick-line',
                        ylabel='GOps',
                        title = '',
                        legend=['Round', '$\Sigma\Delta$', 'Original'],
                        axis=gs1[1, 0],
                        grid=True,
                        )
                ax.set_xticklabels([])
                dbplot(100*np.array([cummean(sd_top1_correct), cummean(sd_top5_correct)]).T, "Score",
                    plot_type=lambda: LinePlot(y_bounds=(0, 100),plot_kwargs=[dict(linewidth=3, color='k'), dict(linewidth=3, color='k', linestyle=':')]),
                    title='',
                    legend=['Round/$\Sigma\Delta$ Top-1', 'Round/$\Sigma\Delta$ Top-5'],
                    ylabel='Cumulative\nPercent Accuracy',
                    xlabel='Frame #',
                    layout='v',
                    axis=gs1[2, 0],
                    grid=True,
                    )
                ax=dbplot((np.arange(1, 20), np.array([round_costs.mean(axis=0), sd_costs.mean(axis=0), full_convnet_cost.mean(axis=0)]).T /1e9), 'Layerwise Computational Costs',
                    ylabel = 'GOps/frame',
                    legend=['Round', '$\Sigma\Delta$', 'Original'],
                    plot_type=lambda: LinePlot(plot_kwargs = dict(marker='.', linewidth=2, markersize=10), y_bounds = (0, None)),
                    axis=gs2[0, 0],
                    grid=True,
                    )
                ax.set_xticklabels([])
                dbplot((np.arange(1, 20), sd_costs.mean(axis=0)/round_costs.mean(axis=0)), 'Ratio',
                    xlabel='Layer #',
                    ylabel='$\Sigma\Delta$:Round\nRatio',
                    title='',
                    plot_type=lambda: LinePlot(plot_kwargs = dict(marker='.', linewidth=2, markersize=10, color='k'), y_bounds = (0, None)),
                    # plot_type=lambda: LinePlot(plot_kwargs=dict(color='gray', linestyle='--')),
                    axis = gs2[1, 0],
                    grid=True,
                    )
            plt.savefig(os.path.join(output_dir, 'epoch-%.3g.pdf' % (minibatch_info.epoch, )))
        f_train_slrc(input_minibatch)
        p()
        print "Epoch {:3.2f}: Scales: {}".format(minibatch_info.epoch, ['%.3g' % float(s) for s in f_get_scales()])

    results = dict(
        current_scales=current_scales,
        round_cost=round_costs.sum(axis=1),
        round_out=round_out,
        sd_cost=sd_costs.sum(axis=1),
        sd_out=sd_out,
        round_guesses=round_guesses,
        round_top1_correct=round_top1_correct,
        round_top5_correct=round_top5_correct,
        sd_guesses=sd_guesses,
        sd_top1_correct=sd_top1_correct,
        sd_top5_correct=sd_top5_correct
        )

    dbplot_hang()
    return results


def get_and_report_scores(comp_cost, output, true_top_1, true_top_k, name):
    guesses = np.argmax(flatten2(output), axis=1)
    top1_correct = guesses==true_top_1
    top5_correct = np.any(guesses[:, None]==true_top_k, axis=1)
    print name + ':\n'+'\n  '.join([
        'Top-1 Score: {:3.2f}%'     .format(top1_correct.mean()*100),
        'Top-5 Score: {:3.2f}%'     .format(top5_correct.mean()*100),
        'Mean Cost: {:3.2f}GOps'.format(comp_cost.mean()/1e9),
        ])
    return guesses, top1_correct, top5_correct


def flatten2(data):
    return data.reshape(data.shape[0], -1)


def argtopk(x, k, axis=-1):
    return np.take(np.argpartition(-x, axis=axis, kth=k), np.arange(k), axis=axis)


def percent_in_top_k(guesses, top_k):
    """
    :param guesses: A (n_samples, ) array
    :param top_k: A (n_samples, k) array indicating the "top k" guesses
    :return: The percent of samples from guesses that are in the top k
    """
    assert guesses.ndim==1
    assert top_k.ndim==2
    assert len(guesses) == len(top_k)
    return 100*np.mean(np.any(guesses[:, None]==top_k))


demo_optimize_conv_scales.add_variant('paper_result', n_epochs=5, comp_weight = 1e-11, learning_rate=0.1, error_loss='L2')

demo_optimize_conv_scales.add_variant('paper_result_mod', n_epochs=5, comp_weight = 2e-12, learning_rate=1., error_loss='L2')


@experiment_function
def scan_lambdas_for_conv_exp():

    results = OrderedDict()
    for comp_weight in [3e-13, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10]:
        results['lambda=%.3g' % (comp_weight, )]=demo_optimize_conv_scales(n_epochs=5, comp_weight = 2e-12, learning_rate=0.1, error_loss='KL')
    return results


if __name__ == '__main__':
    demo_optimize_conv_scales.get_variant('paper_result_mod').run()
