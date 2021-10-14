from collections import defaultdict
import numpy as np
import sys
import cv2
import pandas as pd

label_list = [
    'Bird_spp',
    'Blue_sheep',
    'Glovers_pika',
    'Gray_wolf',
    'Himalaya_marmot',
    'Red_fox',
    'Snow_leopard',
    'Tibetan_snowcock',
    'Upland_Buzzard',
    'White-lipped_deer'
]


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args):
        self.ignore_threshold = args.ignore_threshold
        self.labels = label_list
        self.num_classes = len(self.labels)
        self.results = defaultdict(list)
        self.det_boxes = []
        self.nms_thresh = args.nms_thresh

    def do_nms_for_results(self):
        """Get result boxes."""
        for clsi in self.results:
            dets = self.results[clsi]
            dets = np.array(dets)
            keep_index = self._nms(dets, self.nms_thresh)

            keep_box = [{'category_id': self.labels[int(clsi)],
                         'bbox': list(dets[i][:4].astype(float)),
                         'score': dets[i][4].astype(float)}
                        for i in keep_index]
            self.det_boxes.extend(keep_box)

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indexs = np.where(ovr <= threshold)[0]
            order = order[indexs + 1]
        return reserved_boxes

    def detect(self, outputs, batch, image_shape, config=None):
        """Detect boxes."""
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
                ori_w, ori_h = image_shape
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
                cls_emb = cls_emb.reshape(-1, config.num_classes)
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
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = str(clsi)
                    self.results[coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])


    def draw_boxes_in_image(self, img):
        for i in range(len(self.det_boxes)):
            x = int(self.det_boxes[i]['bbox'][0])
            y = int(self.det_boxes[i]['bbox'][1])
            w = int(self.det_boxes[i]['bbox'][2])
            h = int(self.det_boxes[i]['bbox'][3])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            score = round(self.det_boxes[i]['score'], 3)
            text = self.det_boxes[i]['category_id'] + ', ' + str(score)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        return img
