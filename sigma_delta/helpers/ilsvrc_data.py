import numpy as np
from artemis.general.numpy_helpers import get_rng
from artemis.ml.datasets.ilsvrc import load_ilsvrc_video
from plato.tools.pretrained_networks.vggnet import im2vgginput


def get_vgg_video_splice(video_identifiers, shuffle=False, shuffling_rng = None):

    videos = np.concatenate([load_ilsvrc_video(identifier, size=(224, 224)) for identifier in video_identifiers])
    vgg_mode_videos = im2vgginput(videos)

    if shuffle:
        rng = get_rng(shuffling_rng)
        rng.shuffle(vgg_mode_videos)

    return videos, vgg_mode_videos