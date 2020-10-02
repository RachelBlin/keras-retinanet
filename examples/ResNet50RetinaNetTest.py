# show images inline
#%matplotlib inline

# automatically reload modules when they have changed
#%load_ext autoreload
#%autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import h5py
from PIL import Image

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def detect_image(image_path, label_path, labels_to_names, model):
    image = read_image_bgr(image_path)
    path_split = image_path.split("/")
    name_image = path_split[-1]
    name_split = name_image.split(".")
    number = name_split[0]

    """"# images de Cerema AWP
    # Ouverture du fichier
    mon_fichier = h5py.File('/Users/rblin/Downloads/cerema0.25.hdf5', 'r')

    # Affichage de la structure des dossiers dans le fichier hdf5
    list_elmts = [key for key in mon_fichier['/'].keys()]
        for key in list_elmts:
        print(key)
        print(type(mon_fichier['/'][key]))
        print(mon_fichier['/'][key])
        print([key for key in mon_fichier['/'][key].keys()])

    # Accès aux dossiers et fichiers que l'on veut manipuler
    mon_dataset_train = mon_fichier['train']

    mon_dataset = mon_dataset_train['images']

    image = mon_dataset[30]"""

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        """if label==0:
            content = str(1) + " " + str(int(round(box[0]))) + " " + str(int(round(box[1]))) + " " + str(int(round(box[2] - box[0]))) + " " + str(int(round(box[3] - box[1]))) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()
        elif label==1:
            content = str(15) + " " + str(int(round(box[0]))) + " " + str(int(round(box[1]))) + " " + str(
            int(round(box[2] - box[0]))) + " " + str(int(round(box[3] - box[1]))) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()
        elif label==2:
            content = str(4) + " " + str(int(round(box[0]))) + " " + str(int(round(box[1]))) + " " + str(
            int(round(box[2] - box[0]))) + " " + str(int(round(box[3] - box[1]))) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()
        elif label==3:
            content = str(16) + " " + str(int(round(box[0]))) + " " + str(int(round(box[1]))) + " " + str(
            int(round(box[2] - box[0]))) + " " + str(int(round(box[3] - box[1]))) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()"""
        # YOLO coordinates are x y width height where x and y are the coordinates of the bounding box's
        # center and relative to the width and height of the image
        # In our case the networks returns xmin ymin xmax ymax
        # To get the right format we must convert those coordinates to the following format :
        # (xmin + (xmax-xmin)/2)/width (ymin + (ymax-ymin)/2)/height (xmax-xmin)/width (ymax - ymin)/height

        """if label == 0:
            content = str(1) + " " + str((box[0]+(box[2] - box[0])/2)/500) + " " + str((box[1]+(box[3]-box[1])/2)/500) + " " + str(
                (box[2] - box[0])/500) + " " + str((box[3] - box[1])/500) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()
        elif label == 1:
            content = str(15) + " " + str((box[0]+(box[2] - box[0])/2)/500) + " " + str((box[1]+(box[3]-box[1])/2)/500) + " " + str(
                (box[2] - box[0])/500) + " " + str((box[3] - box[1])/500) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()
        elif label == 2:
            content = str(4) + " " + str((box[0]+(box[2] - box[0])/2)/500) + " " + str((box[1]+(box[3]-box[1])/2)/500) + " " + str(
                (box[2] - box[0])/500) + " " + str((box[3] - box[1])/500) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()
        elif label == 3:
            content = str(16) + " " + str((box[0]+(box[2] - box[0])/2)/500) + " " + str((box[1]+(box[3]-box[1])/2)/500) + " " + str(
                (box[2] - box[0])/500) + " " + str((box[3] - box[1])/500) + "\n"
            label = open(label_path + number + '.txt', 'a')
            label.write(content)
            label.close()"""

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

def detect_image_fusion(image_path1, image_path2, label_path, labels_to_names, model):
    image1 = read_image_bgr(image_path1)
    path_split1 = image_path1.split("/")
    name_image1 = path_split1[-1]
    name_split1 = name_image1.split(".")
    number1 = name_split1[0]

    image2 = read_image_bgr(image_path2)
    path_split2 = image_path2.split("/")
    name_image2 = path_split2[-1]
    name_split2 = name_image2.split(".")
    number2 = name_split2[0]

    # copy to draw on
    draw = image1.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image1 = preprocess_image(image1)
    image1, scale = resize_image(image1)
    image2 = preprocess_image(image2)
    image2, scale = resize_image(image2)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image1, axis=0), np.expand_dims(image2, axis=0)])
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

def detect_image_or_filter_fusion_multimodal(image_path1, image_path2, label_path, labels_to_names, model):
    image1 = read_image_bgr(image_path1)
    path_split1 = image_path1.split("/")
    name_image1 = path_split1[-1]
    name_split1 = name_image1.split(".")
    number1 = name_split1[0]

    image2 = read_image_bgr(image_path2)
    path_split2 = image_path2.split("/")
    name_image2 = path_split2[-1]
    name_split2 = name_image2.split(".")
    number2 = name_split2[0]

    # copy to draw on
    draw = image1.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image1 = preprocess_image(image1)
    image1, scale1 = resize_image(image1)
    image2 = preprocess_image(image2)
    image2, scale2 = resize_image(image2)

    # process image
    start = time.time()
    boxes1, boxes2, scores1, scores2, labels1, labels2 = model.predict_on_batch([np.expand_dims(image1, axis=0), np.expand_dims(image2, axis=0)])
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes1 /= scale1
    boxes2 /= scale2
    for z in range(boxes2.shape[1]):
        if boxes2[0][z][0] >= 0.0:
            # box format is (x1, y1, x2, y2)
            # x_polar = 0.919*x_rgb - 15.4
            # y_polar = 1.04*yrgb - 83
            boxes2[0][z][0] = np.maximum(0, np.minimum(boxes2[0][z][0] * 0.919 - 15.4, 500))
            boxes2[0][z][2] = np.maximum(0, np.minimum(boxes2[0][z][2] * 0.919 - 15.4, 500))
            boxes2[0][z][1] = np.maximum(0, np.minimum(boxes2[0][z][1] * 1.04 - 78, 500))
            boxes2[0][z][3] = np.maximum(0, np.minimum(boxes2[0][z][3] * 1.04 - 78, 500))

    # visualize detections
    for box, score, label in zip(boxes2[0], scores2[0], labels2[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        print(box)

    for box, score, label in zip(boxes1[0], scores1[0], labels1[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label + 10)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
#model_path = "/home/rblin/Documents/keras-retinanet-master/snapshots/resnet50_coco_best_v2.1.0.h5"
#model_path1 = '/home/rblin/Documents/weights/Fusion/no_fusion/Resnet50/Polar/Intensities/resnet50_pascal_07.h5'
#model_path2 = '/home/rblin/Documents/weights/test_rename/temp.h5'
#model_path = '/home/rblin/Documents/weights/Fusion/no_fusion/Resnet50/Polar/Pauli/resnet50_pascal_08.h5'

# load retinanet model
#model_pauli = models.load_model(model_path, backbone_name='resnet50', convert=True)
#model_intensities = models.load_model(model_path1, backbone_name='resnet50', convert=True)
model_path1 = "/home/rblin/Documents/weights/Fusion/no_fusion/Resnet50/Polar/Intensities/resnet50_pascal_07.h5"
model_path2 = "/home/rblin/Documents/weights/test_rename/resnet50_pascal_04.h5"
model_fusion = models.load_model_or_fusion(model_path1, model_path2)

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.load_model(model_path, backbone_name='resnet50', convert=True)

# load label to names mapping for visualization purposes
#labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
#labels_to_names = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 3: 'Person_sitting', 4: 'DontCare', 8: 'Misc', 5: 'Van', 6: 'Tram', 7: 'Truck'}
#labels_to_names = {0: 'bike', 1 : 'car', 2 : 'person', 3 : 'motorbike'}
labels_to_names = {0: 'person', 1: 'bike', 2: 'car', 3: 'motorbike'}
# load image
# image = read_image_bgr('/Users/rblin/Downloads/keras-retinanet-master/examples/000000.png')

#image1 = "/home/rblin/Documents/Databases/PolarLITIS/val_polar/PARAM_POLAR/I04590/0062211.png"
#image2 = "/home/rblin/Documents/Databases/PolarLITIS/val_polar/PARAM_POLAR/Pauli2/0062211.png"

#image1 = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/PARAM_POLAR/I04590/0.png"

#image2 = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/PARAM_POLAR/RGB/frame4065.png"

#label_path = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/LABELS_polar/0.xml"

#label_path = "/home/rblin/Documents/Databases/PolarLITIS/val_polar/LABELS_polar/0062211.xml"

#image1 = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/PARAM_POLAR/I04590/3.png"
#image2 = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/PARAM_POLAR/RGB/frame7272.png"
#label_path = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/LABELS_polar/3.xml"

image1 = "/home/rblin/Documents/Databases/PolarLITIS/diverse_test_polar/PARAM_POLAR/I04590/58.png"
image2 = "/home/rblin/Documents/Databases/PolarLITIS/diverse_test_polar/PARAM_POLAR/RGB/frame10789.png"

label_path = "/home/rblin/Documents/Databases/PolarLITIS/mini_test_polar/LABELS_polar/58.xml"

detect_image_or_filter_fusion_multimodal(image1, image2, label_path, labels_to_names, model_fusion)

#detect_image(image2, label_path, labels_to_names, model_pauli)

#detect_image(image1, label_path, labels_to_names, model_intensities)

"""for image in images:
    path = image_path + image
    detect_image(path, label_path, labels_to_names, model)"""
