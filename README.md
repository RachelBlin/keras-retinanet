This version of RetinaNet is inspired by the one of [fizyr's github](https://github.com/fizyr/keras-retinanet) and has been modified to be trained on polarimetric images. 

The installation steps are the same than fizyr's github, and you can refer to their [README](https://github.com/fizyr/keras-retinanet/blob/master/README.md) for training and testing on other datasets.

## Installation

1) Clone this repository.
2) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
3) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
4) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb) or with [this code](https://github.com/RachelBlin/keras-retinanet/blob/master/python/TestRetinaNet50LRP.py)
In general, inference of the network works as follows:
```python
boxes, scores, labels = model.predict_on_batch(inputs)
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from keras_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')
```

Execution time on NVIDIA Pascal Titan X is roughly 75msec for an image of shape `1000x800x3`.

### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5

# Using the installed script:
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.


## Training
`keras-retinanet` can be trained using [this](https://github.com/RachelBlin/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

The default backbone is `resnet50`. You can change this using the `--backbone=xxx` argument in the running script.
`xxx` can be one of the backbones in resnet models (`resnet50`, `resnet101`, `resnet152`), mobilenet models (`mobilenet128_1.0`, `mobilenet128_0.75`, `mobilenet160_1.0`, etc), densenet models or vgg models. The different options are defined by each model in their corresponding python scripts (`resnet.py`, `mobilenet.py`, etc).

Trained models can't be used directly for inference. To convert a trained model to an inference model, check [here](https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model).

### Usage

For training on [Polar dataset](http://pagesperso.litislab.fr/rblin/databases/)
```shell
# Running directly from the repository:
python keras_retinanet/bin/train.py --epochs number_of_epoch --batch-size batch_size --steps number_of_steps_per_epoch --weights /path/to/weights/for/fine/tuning --snapshot-path /path/to/save/snapshots pascal /path/to/dataset/main/folder/ /relative/path/to/the/train/folder/from/dataset/repository /relative/path/to/the/validation/folder/from/dataset/repository
```

For evaluatiing on [Polar dataset](http://pagesperso.litislab.fr/rblin/databases/)
```shell
# Running directly from the repository:
python keras_retinanet/bin/evaluate.py pascal /path/to/dataset/main/folder/ /relative/path/to/the/test/folder/from/dataset/repository /relative/path/to/the/test/labels/folder/from/dataset/repository /path/to/weights  (--convert-model if needed)
```

## Results

Example output images using `keras-retinanet` are shown below.

### (I0, I45, I90)

<p align="center">
  <img src="https://github.com/RachelBlin/keras-retinanet/blob/master/examples/detection_I.png" alt="Example result of RetinaNet on (I0, I45, I90)"/>
</p>

### (S0, S1, S2)

<p align="center">
     <img src="https://github.com/RachelBlin/keras-retinanet/blob/master/examples/detection_Stokes.png" alt="Example result of RetinaNet on (S0, S1, S2)"/>
</p>  


### (I0, AOP, DOP)

<p align="center">
  <img src="https://github.com/RachelBlin/keras-retinanet/blob/master/examples/detection_Params.png" alt="Example result of RetinaNet on (I0, AOP, DOP)"/>
</p>

### Notes
* This repository requires Keras 2.2.0 or higher.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using OpenCV 3.4.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using Python 2.7 and 3.6.
