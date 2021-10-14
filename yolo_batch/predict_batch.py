# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YoloV3 predict batch."""
import argparse
import os
import sys
import datetime
from collections import defaultdict
import json
import numpy as np
import cv2
import pandas as pd

import mindspore as ms
from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import ConfigYOLOV3DarkNet53
from src.yolo import YOLOV3DarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_datasetv2


class DetectionEngine():
    """Detection engine"""
    def __init__(self, args_engine):
        self.ignore_threshold = args_engine.ignore_threshold
        self.labels = ['Bird_spp', 'Blue_sheep', 'Glovers_pika', 'Gray_wolf', 'Himalaya_marmot', 'Red_fox',
                       'Snow_leopard', 'Tibetan_snowcock', 'Upland_Buzzard', 'White-lipped_deer']
        self.num_classes = len(self.labels)
        self.results = {}  # img_id->class
        self.file_path = ''  # path to save predict result
        self.output_path = args_engine.output_dir
        self.det_boxes = []
        self.nms_thresh = args_engine.nms_thresh
        self.coco_catids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
                            28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
                            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                            81, 82, 84, 85, 86, 87, 88, 89, 90]

    def do_nms_for_results(self):
        """nms result"""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._diou_nms(dets, thresh=0.6)

                keep_box = [{'image_id': int(img_id),
                             'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)}
                            for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, dets, thresh):
        """nms function"""
        # convert xywh -> xmin ymin xmax ymax
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def _diou_nms(self, dets, thresh=0.5):
        """convert xywh -> xmin ymin xmax ymax"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self):
        """write result to json file"""
        try:
            self.file_path = self.output_path + '/predict_image_box.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def write_excel_results(self, excel_results, output_path):
        excel_path = os.path.join(output_path, 'predict_results.xlsx')
        writer = pd.ExcelWriter(excel_path)
        df1 = pd.DataFrame(data=excel_results, columns=['id', 'img_name', 'species', 'score'])
        df1.to_excel(writer, 'predict_results', index=False)
        writer.save()

    def draw_boxes_in_image(self, data, img, img_id, img_name):
        res_list = []
        for i in range(len(data)):
            x = int(data[i]['bbox'][0])
            y = int(data[i]['bbox'][1])
            w = int(data[i]['bbox'][2])
            h = int(data[i]['bbox'][3])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            score = round(data[i]['score'], 3)
            species = self.labels[data[i]['category_id'] - 1]
            text = species + ', ' + str(score)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            res = (int(img_id), img_name, species, score)
            res_list.append(res)
        return img, res_list

    def draw_image(self, img_path):
        excel_results = []
        with open(self.file_path, 'r') as json_file:
            data = json.load(json_file)
        tmp_dict = defaultdict(list)
        for i in range(len(data)):
            img_id = str(data[i].get('image_id'))
            tmp_dict[img_id].append(data[i])

        for key in tmp_dict.keys():
            img_name = key + '.JPG'
            img_file = os.path.join(img_path, img_name)
            ori_image = cv2.imread(img_file, 1)
            img, res_list = self.draw_boxes_in_image(tmp_dict.get(key), ori_image, key, img_name)
            output_img = 'output_' + os.path.basename(img_file).lower()
            cv2.imwrite(os.path.join(self.output_path, output_img), img)
            excel_results.extend(res_list)
        self.write_excel_results(excel_results, self.output_path)

    def detect(self, outputs, batch, image_shape, image_id):
        """post process"""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                conf = conf.reshape(-1)
                cls_argmax = cls_argmax.reshape(-1)

                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                # create all False
                flag = np.random.random(cls_emb.shape) > sys.maxsize
                for i in range(flag.shape[0]):
                    c = cls_argmax[i]
                    flag[i, c] = True
                confidence = cls_emb[flag] * conf
                for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                    if confi < self.ignore_threshold:
                        continue
                    if img_id not in self.results:
                        self.results[img_id] = defaultdict(list)
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = self.coco_catids[clsi]
                    self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore shanshui testing')
    # device related
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: GPU)')
    # dataset related
    parser.add_argument('--data_dir', required=True, type=str, default='', help='image file path')
    parser.add_argument('--output_dir', type=str, default='outputs/predict', help='image file output folder')
    parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')
    parser.add_argument('--group_size', default=1, type=int, help='device num')
    # network related
    parser.add_argument('--pretrained', required=True, default='', type=str,
                        help='model_path, local pretrained model to load')
    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')
    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--ignore_threshold', type=float, default=0.1,
                        help='threshold to throw low quality boxes')

    args, _ = parser.parse_known_args()
    return args


def predict_batch():
    """test method"""
    args = parse_args()

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=False, device_id=devid)

    # logger
    output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(output_dir, rank_id)

    args.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)

    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    config = ConfigYOLOV3DarkNet53()
    args.logger.info('testing shape: {}'.format(config.test_img_shape))
    image_path = os.path.join(args.data_dir, "images")
    image_txt = os.path.join(output_dir, 'predict_image.txt')
    if not os.path.exists(image_txt):
        files = os.listdir(image_path)
        for file in files:
            fullname = os.path.join(image_path, file)
            if fullname.endswith(".JPG"):
                output = open(image_txt, 'a')
                output.write(fullname + '\n')

    ds, data_size = create_yolo_datasetv2(image_path, data_txt=image_txt, batch_size=args.per_batch_size,
                                          max_epoch=1, device_num=args.group_size, rank=rank_id, shuffle=False,
                                          default_config=config)

    args.logger.info('testing shape : %s', config.test_img_shape)
    args.logger.info('totol %d images to eval', data_size)

    network.set_train(False)

    # init detection engine
    config.nms_thresh = args.nms_thresh
    config.ignore_threshold = args.ignore_threshold
    args.logger.info('ignore_threshold: {}'.format(config.ignore_threshold))
    config.output_dir = output_dir
    args.logger.info('result output dir: %s', config.output_dir)
    detection = DetectionEngine(config)

    args.logger.info('Start inference....')
    for i, data in enumerate(ds.create_dict_iterator()):
        image = Tensor(data["image"])

        image_shape = Tensor(data["image_shape"])
        image_id = Tensor(data["img_id"])

        input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
        prediction = network(image, input_shape)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id = image_id.asnumpy()
        image_shape = image_shape.asnumpy()

        detection.detect([output_small, output_me, output_big], args.per_batch_size, image_shape, image_id)
        if i % 1000 == 0:
            args.logger.info('Processing... {:.2f}% '.format(i * args.per_batch_size / data_size * 100))

    args.logger.info('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    args.logger.info('result file path: %s', result_file_path)
    detection.draw_image(image_path)


if __name__ == "__main__":
    predict_batch()

