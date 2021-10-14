# MindSpore YOLOv3-DarkNet53 动物物种检测实践

本文档用于指导大家在GPU环境下完成MindSpore YOLOv3-DarkNet53动物物种检测实践。

> **注意：** 该教程的代码是基于`v1.2.0`及以上版本的MindSpore (https://gitee.com/mindspore/mindspore/tree/r1.2/)运行的。

## 上手指导

### 安装依赖库

* Python库

    ```
    pip install opencv-python flask easydict flask_cors pillow pandas
    ```

* MindSpore (**v1.1.1**)

    为方便快速上手实践，可直接使用MindSpore GPU容器部署环境。使用MindSpore的安装教程请移步至 [MindSpore安装页面](https://gitee.com/mindspore/mindspore/blob/master/README.md).

### 单张推理（yolo_web）
    该项目代码用于部署推理可视化项目，其中pages目录存放前端页面，可在web界面执行单张物种检测任务

##### 推理图片准备
    在web界面上传待推理图片

##### 模型推理
    事先将yolov3预训练模型置于指定目录下（如：./ckpt_path）

##### 部署前端web项目
    可安装nginx进行部署

##### 执行sever.py脚本，启动后台
    ```
    python server.py
    ```

### 批量推理项目（yolo_batch）
    该项目代码用于执行物种批量检测任务

##### 推理数据集准备
    事先将推理任务所需的数据集置于指定文件夹（如：/dataset/eval_data）的images目录下

##### 模型推理
    事先将yolov3预训练模型置于指定目录下（如：./ckpt_path）

##### 执行指令
    ```
    python predict_batch.py --data_dir /dataset/eval_data --ignore_threshold 0.1 --pretrained ./ckpt-path/yolov3.ckpt
    ```

## 许可证

[Apache License 2.0](LICENSE)
