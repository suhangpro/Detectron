from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import math
import random
import json

import numpy as np
import scipy.misc

#from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
#import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
#import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

sys.path.append('build')
import MatterSim

#c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

DATA_ROOT = 'data/'


def list_rooms(data_root=DATA_ROOT):
    dir_rooms = os.path.join(data_root, 'v1', 'scans')
    return [p for p in os.listdir(dir_rooms) if os.path.isdir(os.path.join(dir_rooms, p))]


def list_viewpoints(room, data_root=DATA_ROOT):
    # dir_viewpoints = os.path.join(data_root, 'v1', 'scans', room, 'matterport_skybox_images')
    # return list(set([s.split('_')[0] for s in os.listdir(dir_viewpoints) if s.endswith('.jpg')]))
    room_meta = json.load(open(os.path.join('connectivity', '{}_connectivity.json'.format(room)), 'r'))
    return [str(v['image_id']) for v in room_meta if v['included']]


def parse_args():
    parser = argparse.ArgumentParser(description='Detection on Matterport3D')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'),
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
        type=str
    )
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)
    parser.add_argument('--vfov', default=60, type=int)
    parser.add_argument('--heading-mode', default='auto', choices=('auto','free'))
    parser.add_argument('--elevation-mode', default='lock', choices=('auto', 'free', 'lock'))
    parser.add_argument('--room', default='17DRP5sb8fy', type=str)
    parser.add_argument('--start', default='902e65564f81489687878425d9b3cb55', type=str)
    parser.add_argument('--output-dir', default='/tmp/infer_matterport', type=str)
    return parser.parse_args()


def main(args):
    vfov, width, height = math.radians(args.vfov), args.width, args.height
    hfov = vfov * width / height
    direction_text_color = [230, 40, 40]

    cv2.namedWindow('Matterport3D')
    sim = MatterSim.Simulator()
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(vfov)
    sim.setDiscretizedViewingAngles(False)
    sim.init()
    # sim.newEpisode('2t7WUuJeko7', '1e6b606b44df4a6086c0f97e826d4d15', 0, 0)
    # sim.newEpisode('17DRP5sb8fy', '902e65564f81489687878425d9b3cb55', 0, 0)

    roomId, viewId = args.room, args.start
    sim.newEpisode(roomId, viewId, 0, 0)
    rooms = set(list_rooms())

    # putting these before the newEpisode call leads to segmentation fault
    from caffe2.python import workspace
    import core.test_engine as infer_engine
    import utils.c2 as c2_utils
    c2_utils.import_detectron_ops()

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)

    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    angle_delta_lg = 30 * math.pi / 180
    angle_delta_sm = 5 * math.pi / 180

    while True:
        location = 0
        heading = 0
        elevation = 0
        state = sim.getState()
        locations = state.navigableLocations
        loc = locations[0]
        bgr = state.rgb
        logger.info('Pose: {} - {} ({},{},{}) - {}/{}'.format(
            roomId, loc.viewpointId, loc.point[0], loc.point[1], loc.point[2], state.heading, state.elevation))
        timers = defaultdict(Timer)
        t = time.time()

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, bgr, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        im = vis_utils.vis_one_image_opencv(
            bgr.astype(np.uint8).copy(), 
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            show_class=True,
            show_box=True,
            thresh=0.7,
            kp_thresh=2
        )

        im_with_directions = im.copy()
        for idx, loc in enumerate(locations[1:]):
            # Draw actions on the screen
            fontScale = 3.0/loc.rel_distance
            x = int(width/2 + loc.rel_heading / hfov * width)
            y = int(height/2 - loc.rel_elevation / vfov * height)
            cv2.putText(im_with_directions, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, direction_text_color, thickness=3)
        cv2.imshow('Matterport3D', im_with_directions)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif ord('1') <= k <= ord('9'):
            location = k - ord('0')
            if location >= len(locations):
                location = 0
            else:
                if args.heading_mode == 'auto':
                    heading = locations[location].rel_heading
                if args.elevation_mode == 'auto':
                    elevation = locations[location].rel_elevation
                elif args.elevation_mode == 'lock':
                    elevation = - state.elevation
        elif k == 81:
            heading = -angle_delta_lg
        elif k == 82:
             elevation = angle_delta_lg
        elif k == 83:
             heading = angle_delta_lg
        elif k == 84:
             elevation = -angle_delta_lg
        elif k == ord('a'):
            heading = -angle_delta_sm
        elif k == ord('w'):
             elevation = angle_delta_sm
        elif k == ord('d'):
             heading = angle_delta_sm
        elif k == ord('s'):
             elevation = -angle_delta_sm
        elif k == ord('b'):
            heading = math.pi
        elif k == ord('p'):
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            cv2.imwrite(os.path.join(args.output_dir, '{}_{}_{}_{}_{}.jpg'.format(
                roomId, viewId, state.heading, state.elevation, args.vfov)), im)
        elif k == ord('o'):
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            cv2.imwrite(os.path.join(args.output_dir, '{}_{}_{}_{}_{}_raw.jpg'.format(
                roomId, viewId, state.heading, state.elevation, args.vfov)), bgr)
        elif k == ord('n'):
            roomId = random.sample(rooms.difference({roomId}), 1)[0]
            viewId = random.sample(list_viewpoints(roomId), 1)[0]
            sim.newEpisode(roomId, viewId, 0, 0)
            continue
        elif k == ord('r'):
            viewId = random.sample(list_viewpoints(roomId), 1)[0]
            sim.newEpisode(roomId, viewId, 0, 0)
            continue

        sim.makeAction(location, heading, elevation)


if __name__ == '__main__':
    args = parse_args()
    main(args)

