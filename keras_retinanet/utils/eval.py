"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os

import cv2


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_detections_fusion(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image[0] = image[0].transpose((2, 0, 1))
            image[1] = image[1].transpose((2, 0, 1))

        # run network
        # empty_image = np.zeros(image[0].shape)
        # boxes, scores, labels = model.predict_on_batch([np.expand_dims(empty_image, axis=0), np.expand_dims(image[1], axis=0)])[:3]
        boxes, scores, labels = model.predict_on_batch(
            [np.expand_dims(image[0], axis=0), np.expand_dims(image[1], axis=0)])[:3]
        # correct boxes for image scale
        boxes /= scale[0]

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_detections_or_fusion(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image[0] = image[0].transpose((2, 0, 1))
            image[1] = image[1].transpose((2, 0, 1))

        # run network
        # empty_image = np.zeros(image[0].shape)
        # boxes, scores, labels = model.predict_on_batch([np.expand_dims(image[0], axis=0), np.expand_dims(empty_image, axis=0)])[:3]
        boxes1, boxes2, scores1, scores2, labels1, labels2 = model.predict_on_batch(
            [np.expand_dims(image[0], axis=0), np.expand_dims(image[1], axis=0)])[:6]

        # correct boxes for image scale
        boxes1 /= scale[0]
        boxes2 /= scale[1]

        # select indices which have a score above the threshold
        indices1 = np.where(scores1[0, :] > score_threshold)[0]
        indices2 = np.where(scores2[0, :] > score_threshold)[0]

        # select those scores
        scores1 = scores1[0][indices1]
        scores2 = scores2[0][indices2]

        # find the order with which to sort the scores
        scores_sort1 = np.argsort(-scores1)[:max_detections]
        scores_sort2 = np.argsort(-scores2)[:max_detections]

        image_boxes1 = boxes1[0, indices1[scores_sort1], :]
        image_scores1 = scores1[scores_sort1]
        image_labels1 = labels1[0, indices1[scores_sort1]]
        image_boxes2 = boxes2[0, indices2[scores_sort2], :]
        image_scores2 = scores2[scores_sort2]
        image_labels2 = labels2[0, indices2[scores_sort2]]

        # Initialize the detected boxes to the ones of first modality
        image_boxes = image_boxes1
        image_scores = image_scores1
        image_labels = image_labels1
        for j in range(image_boxes1.shape[0]):
            box1_temp = image_boxes1[j]
            box_sup_temp_index = None
            score_temp = image_scores[j]
            for k in range(image_boxes2.shape[0]):
                box2_temp = image_boxes2[k]
                if intersection_over_union(box1_temp, box2_temp) > 0.89 and image_labels[j] == image_labels2[k]:
                    if image_scores2[k] > score_temp:
                        box_sup_temp_index = k
                        score_temp = image_scores2[k]
            if box_sup_temp_index is not None:
                image_boxes[j] = image_boxes2[box_sup_temp_index]
                image_scores[j] = image_scores2[box_sup_temp_index]

        for l in range(image_boxes2.shape[0]):
            flag = 1
            for m in range(image_boxes1.shape[0]):
                if intersection_over_union(image_boxes2[l], image_boxes1[m]) >= 0.05 or image_scores2[l] <= 0.05:
                    flag = 0
            if flag == 1:
                image_boxes = np.append(image_boxes, [image_boxes2[l]], axis=0)
                image_scores = np.append(image_scores, [image_scores2[l]], axis=0)
                image_labels = np.append(image_labels, [image_labels2[l]], axis=0)

        # select detections
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_detections_or_fusion_multimodal(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image[0] = image[0].transpose((2, 0, 1))
            image[1] = image[1].transpose((2, 0, 1))

        # run network
        #empty_image = np.zeros(image[0].shape)
        #boxes1, boxes2, scores1, scores2, labels1, labels2 = model.predict_on_batch([np.expand_dims(image[0], axis=0), np.expand_dims(empty_image, axis=0)])[:6]
        boxes1, boxes2, scores1, scores2, labels1, labels2 = model.predict_on_batch([np.expand_dims(image[0], axis=0), np.expand_dims(image[1], axis=0)])[:6]

        # correct boxes for image scale
        boxes1 /= scale[0]
        boxes2 /= scale[1]
        for z in range(boxes2.shape[1]):
                    if boxes2[0][z][0] >= 0.0:
                        # box format is (x1, y1, x2, y2)
                        # x_polar = 0.919*x_rgb - 15.4
                        # y_polar = 1.04*yrgb - 83
                        boxes2[0][z][0] = np.maximum(0, np.minimum(boxes2[0][z][0] * 0.919 - 15.4, 500))
                        boxes2[0][z][2] = np.maximum(0, np.minimum(boxes2[0][z][2] * 0.919 - 15.4, 500))
                        boxes2[0][z][1] = np.maximum(0, np.minimum(boxes2[0][z][1] * 1.04 - 78, 500))
                        boxes2[0][z][3] = np.maximum(0, np.minimum(boxes2[0][z][3] * 1.04 - 78, 500))

        # select indices which have a score above the threshold
        indices1 = np.where(scores1[0, :] > score_threshold)[0]
        indices2 = np.where(scores2[0, :] > score_threshold)[0]

        # select those scores
        scores1 = scores1[0][indices1]
        scores2 = scores2[0][indices2]

        # find the order with which to sort the scores
        scores_sort1 = np.argsort(-scores1)[:max_detections]
        scores_sort2 = np.argsort(-scores2)[:max_detections]

        image_boxes1 = boxes1[0, indices1[scores_sort1], :]
        image_scores1 = scores1[scores_sort1]
        image_labels1 = labels1[0, indices1[scores_sort1]]
        image_boxes2 = boxes2[0, indices2[scores_sort2], :]
        image_scores2 = scores2[scores_sort2]
        image_labels2 = labels2[0, indices2[scores_sort2]]

        # Initialize the detected boxes to the ones of first modality
        image_boxes = image_boxes1
        image_scores = image_scores1
        image_labels = image_labels1
        for j in range(image_boxes1.shape[0]):
            box1_temp = image_boxes1[j]
            box_sup_temp_index = None
            score_temp = image_scores[j]
            for k in range(image_boxes2.shape[0]):
                box2_temp = image_boxes2[k]
                if intersection_over_union(box1_temp, box2_temp) > 0.89 and image_labels[j] == image_labels2[k]:
                    if image_scores2[k] > score_temp:
                        box_sup_temp_index = k
                        score_temp = image_scores2[k]
            if box_sup_temp_index is not None:
                image_boxes[j] = image_boxes2[box_sup_temp_index]
                image_scores[j] = image_scores2[box_sup_temp_index]

        for l in range(image_boxes2.shape[0]):
            flag = 1
            for m in range(image_boxes1.shape[0]):
                if intersection_over_union(image_boxes2[l], image_boxes1[m]) >= 0.05 or image_scores2[l] <= 0.5:
                        flag = 0
            if flag == 1:
                image_boxes = np.append(image_boxes, [image_boxes2[l]], axis=0)
                image_scores = np.append(image_scores, [image_scores2[l]], axis=0)
                image_labels = np.append(image_labels, [image_labels2[l]], axis=0)
                print(image_boxes2[l], image_scores2[l], image_labels2[l], i)
                print("************************************************")

        # select detections
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_detections_and_fusion(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image[0] = image[0].transpose((2, 0, 1))
            image[1] = image[1].transpose((2, 0, 1))

        # run network
        # empty_image = np.zeros(image[0].shape)
        # boxes, scores, labels = model.predict_on_batch([np.expand_dims(image[0], axis=0), np.expand_dims(empty_image, axis=0)])[:3]
        boxes1, boxes2, scores1, scores2, labels1, labels2 = model.predict_on_batch(
            [np.expand_dims(image[0], axis=0), np.expand_dims(image[1], axis=0)])[:6]

        # correct boxes for image scale
        boxes1 /= scale[0]
        boxes2 /= scale[1]

        # select indices which have a score above the threshold
        indices1 = np.where(scores1[0, :] > score_threshold)[0]
        indices2 = np.where(scores2[0, :] > score_threshold)[0]

        # select those scores
        scores1 = scores1[0][indices1]
        scores2 = scores2[0][indices2]

        # find the order with which to sort the scores
        scores_sort1 = np.argsort(-scores1)[:max_detections]
        scores_sort2 = np.argsort(-scores2)[:max_detections]

        image_boxes1 = boxes1[0, indices1[scores_sort1], :]
        image_scores1 = scores1[scores_sort1]
        image_labels1 = labels1[0, indices1[scores_sort1]]
        image_boxes2 = boxes2[0, indices2[scores_sort2], :]
        image_scores2 = scores2[scores_sort2]
        image_labels2 = labels2[0, indices2[scores_sort2]]

        # Initialize the detected boxes to the ones of first modality
        image_boxes = np.empty((0, 4), dtype=float)
        image_scores = np.empty((0,), dtype=float)
        image_labels = np.empty((0,), dtype=int)
        for j in range(image_boxes1.shape[0]):
            IOU_temp = 0.0
            index_temp = None
            flag = 0
            for k in range(image_boxes2.shape[0]):
                if intersection_over_union(image_boxes1[j], image_boxes2[k]) > 0.55 and image_labels1[j] == \
                        image_labels2[k] and intersection_over_union(image_boxes1[j], image_boxes2[k]) > IOU_temp:
                    IOU_temp = intersection_over_union(image_boxes1[j], image_boxes2[k])
                    if image_scores2[k] > image_scores1[j]:
                        index_temp = k
                        flag = 2
                    else:
                        index_temp = j
                        flag = 1

            if index_temp is not None:
                if flag == 2:
                    image_boxes = np.append(image_boxes, [image_boxes2[index_temp]], axis=0)
                    image_scores = np.append(image_scores, [image_scores2[index_temp]], axis=0)
                    image_labels = np.append(image_labels, [image_labels2[index_temp]], axis=0)
                elif flag == 1:
                    image_boxes = np.append(image_boxes, [image_boxes1[index_temp]], axis=0)
                    image_scores = np.append(image_scores, [image_scores1[index_temp]], axis=0)
                    image_labels = np.append(image_labels, [image_labels1[index_temp]], axis=0)

        # select detections
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_detections_and_fusion_multimodal(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image[0] = image[0].transpose((2, 0, 1))
            image[1] = image[1].transpose((2, 0, 1))

        # run network
        # empty_image = np.zeros(image[0].shape)
        # boxes, scores, labels = model.predict_on_batch([np.expand_dims(image[0], axis=0), np.expand_dims(empty_image, axis=0)])[:3]
        boxes1, boxes2, scores1, scores2, labels1, labels2 = model.predict_on_batch(
            [np.expand_dims(image[0], axis=0), np.expand_dims(image[1], axis=0)])[:6]

        # correct boxes for image scale
        boxes1 /= scale[0]
        boxes2 /= scale[1]
        for z in range(boxes2.shape[1]):
            if boxes2[0][z][0] >= 0.0:
                # box format is (x1, y1, x2, y2)
                # x_polar = 0.919*x_rgb - 15.4
                # y_polar = 1.04*yrgb - 83
                boxes2[0][z][0] = np.maximum(0, np.minimum(boxes2[0][z][0] * 0.919 - 15.4, 500))
                boxes2[0][z][2] = np.maximum(0, np.minimum(boxes2[0][z][2] * 0.919 - 15.4, 500))
                boxes2[0][z][1] = np.maximum(0, np.minimum(boxes2[0][z][1] * 1.04 - 78, 500))
                boxes2[0][z][3] = np.maximum(0, np.minimum(boxes2[0][z][3] * 1.04 - 78, 500))

        # select indices which have a score above the threshold
        indices1 = np.where(scores1[0, :] > score_threshold)[0]
        indices2 = np.where(scores2[0, :] > score_threshold)[0]

        # select those scores
        scores1 = scores1[0][indices1]
        scores2 = scores2[0][indices2]

        # find the order with which to sort the scores
        scores_sort1 = np.argsort(-scores1)[:max_detections]
        scores_sort2 = np.argsort(-scores2)[:max_detections]

        image_boxes1 = boxes1[0, indices1[scores_sort1], :]
        image_scores1 = scores1[scores_sort1]
        image_labels1 = labels1[0, indices1[scores_sort1]]
        image_boxes2 = boxes2[0, indices2[scores_sort2], :]
        image_scores2 = scores2[scores_sort2]
        image_labels2 = labels2[0, indices2[scores_sort2]]

        # Initialize the detected boxes to the ones of first modality
        image_boxes = np.empty((0, 4), dtype=float)
        image_scores = np.empty((0,), dtype=float)
        image_labels = np.empty((0,), dtype=int)
        for j in range(image_boxes1.shape[0]):
            IOU_temp = 0.0
            index_temp = None
            flag = 0
            for k in range(image_boxes2.shape[0]):
                if intersection_over_union(image_boxes1[j], image_boxes2[k]) > 0.55 and image_labels1[j] == \
                        image_labels2[k] and intersection_over_union(image_boxes1[j], image_boxes2[k]) > IOU_temp:
                    IOU_temp = intersection_over_union(image_boxes1[j], image_boxes2[k])
                    if image_scores2[k] > image_scores1[j]:
                        index_temp = k
                        flag = 2
                    else:
                        index_temp = j
                        flag = 1

            if index_temp is not None:
                if flag == 2:
                    image_boxes = np.append(image_boxes, [image_boxes2[index_temp]], axis=0)
                    image_scores = np.append(image_scores, [image_scores2[index_temp]], axis=0)
                    image_labels = np.append(image_labels, [image_labels2[index_temp]], axis=0)
                elif flag == 1:
                    image_boxes = np.append(image_boxes, [image_boxes1[index_temp]], axis=0)
                    image_scores = np.append(image_scores, [image_scores1[index_temp]], axis=0)
                    image_labels = np.append(image_labels, [image_labels1[index_temp]], axis=0)

        # select detections
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate_fusion(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections_fusion(generator, model, score_threshold=score_threshold,
                                            max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate_or_fusion(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections_or_fusion(generator, model, score_threshold=score_threshold,
                                               max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate_or_fusion_multimodal(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations
    all_detections = _get_detections_or_fusion_multimodal(generator, model, score_threshold=score_threshold,
                                                          max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate_and_fusion(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections_and_fusion(generator, model, score_threshold=score_threshold,
                                                max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate_and_fusion_multimodal(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections_and_fusion_multimodal(generator, model, score_threshold=score_threshold,
                                                           max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def intersection_over_union(box1, box2):
    # First verify if there is an intersection of the two bounding boxes
    # For this

    # First compute the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    # Once we have the coordinates, compute the area of the intersection rectangle
    intersection_rectangle = (x1 - x2) * (y1 - y2)

    # Then compute the area of each of the two bounding boxes
    box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # The intersection over union is computed as follows :
    # area of intersection rectangle / (area of bbox 1 + area of bbox 2 - area of intersection reactangle)
    iou = intersection_rectangle / float(box1 + box2 - intersection_rectangle)

    # return the intersection over union value
    return iou
