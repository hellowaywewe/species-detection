import base64
import os
from io import BytesIO

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
    img = img.transpose(2, 0, 1)
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
    pretrained_ckpt = './ckpt-path/yolo_web.ckpt'
    if not os.path.exists(pretrained_ckpt):
        err_msg = "The yolo_web.ckpt file does not exist!"
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

    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    print('Start inference....')
    network.set_train(False)
    ori_image = np.array(json.loads(instance['data']), dtype=instance['dtype'])
    image, image_shape = data_preprocess(ori_image, config)
    prediction = network(Tensor(image.reshape(1, 3, 416, 416), ms.float32), input_shape)
    output_big, output_me, output_small = prediction
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()

    per_batch_size = 1
    detection.detect([output_small, output_me, output_big], per_batch_size,
                     image_shape, config)
    detection.do_nms_for_results()
    out_img = detection.draw_boxes_in_image(ori_image)

    # for i in range(len(detection.det_boxes)):
    #     print("x: ", detection.det_boxes[i]['bbox'][0])
    #     print("y: ", detection.det_boxes[i]['bbox'][1])
    #     print("h: ", detection.det_boxes[i]['bbox'][2])
    #     print("w: ", detection.det_boxes[i]['bbox'][3])
    #     print("score: ", round(detection.det_boxes[i]['score'], 3))
    #     print("category: ", detection.det_boxes[i]['category_id'])

    det_boxes = detection.det_boxes
    if not len(det_boxes):
        err_msg = "抱歉！未检测到任何种类，无法标注。"
        return {"status": 1, "err_msg": err_msg}
    max_det = max(det_boxes, key=lambda k: k['score'])
    max_score = max_det['score']
    category = det_boxes[det_boxes.index(max_det)]['category_id']

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
