#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single video or all videos with a certain extension
(e.g., mp4) recursively in a folder.
"""

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

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/detect_video)',
        default='/tmp/detect_video',
        type=str
    )
    parser.add_argument(
        '--sample-fps',
        dest='sample_fps',
        help='sampling fps (default: native video fps)',
        default=-1,
        type=int
    )
    parser.add_argument(
        '--sample-offset',
        dest='sample_offset',
        help='sampling starting point offset (default: 0)',
        default=0,
        type=int
    )
    parser.add_argument(
        '--video-ext',
        dest='video_ext',
        help='video file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        '--subdir-each-video',
        dest='subdir_each_video',
        help='create innermost subfolders for each video (default: False)',
        action='store_true'
    )
    parser.add_argument(
        '--skip-viz',
        dest='skip_viz',
        help='skip visualizing detections (default: False)',
        action='store_true'
    )
    parser.add_argument(
        '--bbox-only',
        dest='bbox_only',
        help='skip predicting masks (default: False)',
        action='store_true'
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection threshold (default: 0.7)',
        default=0.7,
        type=float
    )
    parser.add_argument(
        'video_or_folder', help='path to video or folder of videos', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def infer_folder(video_or_folder, model, output_dir,
                 video_ext='mp4',
                 subdir_each_video=True,
                 skip_viz=False,
                 threshold=0.7,
                 infer_sample_fps=-1,
                 infer_sample_offset=0,
                 logger=None):
    video_or_folder = video_or_folder.rstrip('/')
    if os.path.isdir(video_or_folder):
        output_dir = os.path.join(output_dir, os.path.basename(video_or_folder))
        video_list = glob.iglob(video_or_folder + '/*.' + video_ext)
        subdir_list = glob.iglob(video_or_folder + '/*/')
        for subdir in subdir_list:
            infer_folder(subdir, model, output_dir, video_ext, subdir_each_video, skip_viz, threshold,
                         infer_sample_fps, infer_sample_offset, logger)
    else:
        video_list = [video_or_folder]

    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    for video_path in video_list:
        if subdir_each_video:
            curr_output_dir = os.path.join(output_dir, os.path.basename(video_path).rstrip('.' + video_ext))
        else:
            curr_output_dir = output_dir
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        curr_detections_txt = os.path.join(curr_output_dir, 'detections.txt')
        f = open(curr_detections_txt, 'a+')

        if not skip_viz:
            curr_viz_dir = os.path.join(curr_output_dir, 'viz')
            if not os.path.exists(curr_viz_dir):
                os.makedirs(curr_viz_dir)

        cap = cv2.VideoCapture(str(video_path))
        video_width = cap.get(3)  # CAP_PROP_FRAME_WIDTH = 3
        video_height = cap.get(4)  # CAP_PROP_FRAME_HEIGHT = 4
        video_fps = cap.get(5)  # CAP_PROP_FPS = 5
        video_total_frames = cap.get(7)  # CAP_PROP_FRAME_COUNT = 7
        stride = round(float(video_fps) / infer_sample_fps) if infer_sample_fps > 0 else 1
        frame_id = infer_sample_offset
        while True:
            try:
                cap.set(1, frame_id)  # CAP_PROP_POS_FRAMES = 1
                ret, im = cap.read()
                if not ret:
                    break
                logger.info('Processing {} -> {} (frame #{})'.format(video_path, curr_output_dir, frame_id))
                timers = defaultdict(Timer)
                t = time.time()
                with c2_utils.NamedCudaScope(0):
                    cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                        model, im, None, timers=timers
                    )
                logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                for k, v in timers.items():
                    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

                detections = []
                for c, bboxes in enumerate(cls_boxes):
                    detections_c = [' '.join([str(int(v)) for v in [
                        frame_id, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], c,
                        video_width, video_height, video_total_frames, video_fps]]) + '\n'
                                       for bbox in bboxes if bbox[4] >= threshold]
                    detections.extend(detections_c)
                f.writelines(detections)

                if not skip_viz:
                    vis_utils.vis_one_image(
                        im[:, :, ::-1],  # BGR -> RGB for visualization
                        'frame_{:06}'.format(frame_id),
                        curr_viz_dir,
                        cls_boxes,
                        cls_segms,
                        cls_keyps,
                        dataset=dummy_coco_dataset,
                        box_alpha=0.3,
                        show_class=True,
                        thresh=threshold,
                        kp_thresh=2,
                        ext='png'
                    )
                frame_id += stride
            except:
                break

        f.close()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    cfg.MODEL.MASK_ON = not (args.skip_viz or args.bbox_only)
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()

    infer_folder(args.video_or_folder, model, args.output_dir, args.video_ext,
                 subdir_each_video=args.subdir_each_video,
                 skip_viz=args.skip_viz,
                 threshold=args.thresh,
                 infer_sample_fps=args.sample_fps,
                 infer_sample_offset=args.sample_offset,
                 logger=logger)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
