import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import json
import argparse

# SAL real-time deadline harness: when SAL_DEADLINE_FPS is set the
# framework injects a directory containing deadline_iterator.py via
# SAL_RUNTIME_PATH. The import is lazy (inside image_stream) so an
# unmodified DROID env without these vars set works exactly as before.
_SAL_DEADLINE_FPS = os.environ.get("SAL_DEADLINE_FPS")
_SAL_DROP_LOG_PATH = os.environ.get("SAL_DROP_LOG_PATH")


def _load_deadline_iterator():
    """Import DeadlineIterator from SAL_RUNTIME_PATH on demand."""
    runtime_path = os.environ.get("SAL_RUNTIME_PATH")
    if runtime_path and runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    from deadline_iterator import DeadlineIterator  # noqa: E402
    return DeadlineIterator

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride, indices=None):
    """ image generator

    When ``indices`` is None and ``SAL_DEADLINE_FPS`` is set, the image
    list is wrapped in a ``DeadlineIterator`` that silently skips frames
    whose wall-clock deadline has passed. When ``indices`` is provided
    (non-None), the list is filtered to only those positions and no
    deadline check is applied — used by ``droid.terminate(...)`` to
    replay only the frames the live loop actually saw.
    """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    if indices is not None:
        image_list = [image_list[i] for i in indices]
    elif _SAL_DEADLINE_FPS:
        DeadlineIterator = _load_deadline_iterator()
        warmup_frames = int(os.environ.get("SAL_DEADLINE_WARMUP_FRAMES", 0) or 0)
        queue_size = int(os.environ.get("SAL_DEADLINE_QUEUE_SIZE", 1) or 1)
        drop_policy = os.environ.get("SAL_DEADLINE_DROP_POLICY", "drop_oldest") or "drop_oldest"
        image_list = DeadlineIterator(
            image_list,
            float(_SAL_DEADLINE_FPS),
            warmup_frames=warmup_frames,
            queue_size=queue_size,
            drop_policy=drop_policy,
        )

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    # If SAL's deadline harness was active, the live loop wrote a JSON
    # log of which dataset indices were actually processed. Replay only
    # those in droid.terminate so global bundle adjustment doesn't
    # reintroduce frames the SLAM never saw live.
    survivors = None
    if _SAL_DEADLINE_FPS and _SAL_DROP_LOG_PATH and os.path.exists(_SAL_DROP_LOG_PATH):
        try:
            with open(_SAL_DROP_LOG_PATH) as _f:
                survivors = json.load(_f).get("survivors")
        except (OSError, json.JSONDecodeError):
            survivors = None

    traj_est = droid.terminate(
        image_stream(args.imagedir, args.calib, args.stride, indices=survivors)
    )

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
