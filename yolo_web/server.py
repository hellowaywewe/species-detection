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

import base64
import os
from io import BytesIO
import random

from PIL import Image
from flask import Flask, request, jsonify, make_response
import json
from easydict import EasyDict as edict
import numpy as np
import cv2
from flask_cors import CORS, cross_origin

from src.transforms import _reshape_data
from src.yolo import YOLOV3DarkNet53
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.config import ConfigYOLOV3DarkNet53
from mindspore import Tensor
import mindspore as ms
from src.detection import DetectionEngine


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20*1024*1024
CORS(app, supports_credentials=True)


@app.route('/')
@cross_origin(supports_credentials=True)
def hello_world():
    return 'index.html'


def data_preprocess(img, config):
    img, ori_image_shape = _reshape_data(img, config.test_img_shape)
    img = img.transpose(2, 0, 1).copy()
    return img, ori_image_shape


def base64tonumpy(img_base64):
    np_arr = np.fromstring(img_base64, np.uint8)
    img = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
    return np.array(img)


def numpy2base64(img_np):
    img_np = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    output_buffer = BytesIO()
    img_np.save(output_buffer, format="JPEG")
    byte_data = output_buffer.getvalue()
    base64_str = "data:image/jpeg;base64," + str(base64.b64encode(byte_data), encoding="utf-8")
    return base64_str


def yolov3_predict(instance, strategy):
    network = YOLOV3DarkNet53(is_training=False)

    pretrained_ckpt = './ckpt-path/yolov3.ckpt'
    if not os.path.exists(pretrained_ckpt):
        err_msg = "抱歉！yolov3预训练模型不存在!"
        return {"status": 1, "err_msg": err_msg}
    param_dict = load_checkpoint(pretrained_ckpt)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    load_param_into_net(network, param_dict_new)

    config = ConfigYOLOV3DarkNet53()

    # init detection engine
    args = edict()
    args.ignore_threshold = 0.1
    args.nms_thresh = 0.5
    detection = DetectionEngine(args)

    print('Start inference....')
    network.set_train(False)
    ori_image = np.array(json.loads(instance['data']), dtype=instance['dtype'])
    cvt_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image, image_shape = data_preprocess(cvt_image, config)
    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    prediction = network(Tensor(image.reshape(1, 3, 416, 416), ms.float32), input_shape)
    output_big, output_me, output_small = prediction
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()

    image_id = random.randint(0, 100)
    args.per_batch_size = 1
    detection.detect([output_small, output_me, output_big],
                     batch=args.per_batch_size, image_shape=image_shape, image_id=image_id)
    detection.do_nms_for_results()
    out_img = detection.draw_boxes_in_image(ori_image)

    det_boxes = detection.det_boxes
    if not len(det_boxes):
        err_msg = "抱歉！未检测到任何种类，无法标注。"
        return {"status": 1, "err_msg": err_msg}
    max_det = max(det_boxes, key=lambda k: k['score'])
    max_score = max_det['score']
    category = detection.labels[det_boxes[det_boxes.index(max_det)]['category_id'] - 1]

    res = {
        "status": 0,
        "instance": {
            "boxes_num": len(det_boxes),
            "max_score": round(max_score, 3),
            "category": category,
            "data": numpy2base64(out_img)
        }
    }
    return res


@app.route('/predict', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict():
    json_data = json.loads(request.data)
    data = base64.b64decode(json_data['data'])
    image = base64tonumpy(data)
    strategy = 'TOP1'
    instance = {
            'shape': list(image.shape),
            'dtype': image.dtype.name,
            'data': json.dumps(image.tolist())
    }
    res = yolov3_predict(instance, strategy)
    response = make_response(jsonify(res))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Method'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return jsonify(res)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
