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

import keras
from .. import initializers
from .. import layers

import tensorflow as tf

import numpy as np


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

def __create_pyramid_features2(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced2')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled2')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P52')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced2')(C4)
    P4           = keras.layers.Add(name='P4_merged2')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled2')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P42')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced2')(C3)
    P3 = keras.layers.Add(name='P3_merged2')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P32')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P62')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu2')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P72')(P7)

    return [P3, P4, P5, P6, P7]


class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, num_anchors, regression_name='regression_submodel', classification_name='classification_submodel', reg_name='regression', class_name='classification'):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        (reg_name, default_regression_model(num_anchors, name=regression_name)),
        (class_name, default_classification_model(num_classes, num_anchors, name=classification_name))
    ]

def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)

def __build_anchors2(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors2_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors2')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors             = 9,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    regression_name         = 'regression_submodel',
    classification_name     = 'classification_submodel',
    reg_name                = 'regression',
    class_name              = 'classification',
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """
    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors, regression_name, classification_name, reg_name, class_name)

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    anchor_parameters     = AnchorParameters.default,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters     : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model is None:
        model = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = __build_anchors(anchor_parameters, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    outputs = detections

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)

def retinanet_bbox_fusion(
    model1                = None,
    model2                = None,
    anchor_parameters     = AnchorParameters.default,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    **kwargs
):
    """ Construct two RetinaNet models on top of a backbone, one for each modality and adds convenience functions to output boxes directly.

    This model uses two RetinaNet models, each one specialized on a modality. Predictions are made for each model and the obtained results are
    fused before being filtered in a final step.

    Args
        model1                : RetinaNet model from first modality to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        model2                : RetinaNet model from second modality to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters     : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model1 is None:
        model1 = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    if model2 is None:
        model2 = retinanet(num_anchors=anchor_parameters.num_anchors(),
                           create_pyramid_features=__create_pyramid_features2,
                           regression_name='regression_submodel2',
                           classification_name='classification_submodel2',
                           reg_name='regression2',
                           class_name='classification2',
                           name='retinanet2',
                           **kwargs)

    """if model2 is None:
        model2 = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)"""

    # compute the anchors for both modalities
    features1 = [model1.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors1  = __build_anchors(anchor_parameters, features1)
    features2 = [model2.get_layer(p_name).output for p_name in ['P32', 'P42', 'P52', 'P62', 'P72']]
    anchors2 = __build_anchors2(anchor_parameters, features2)

    # we expect the anchors, regression and classification values as first output for both modalities
    regression1     = model1.outputs[0]
    classification1 = model1.outputs[1]
    regression2 = model2.outputs[0]
    classification2 = model2.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other1 = model1.outputs[2:]
    other2 = model2.outputs[2:]

    # apply predicted regression to anchors for each modality
    boxes1 = layers.RegressBoxes(name='boxes1')([anchors1, regression1])
    boxes1 = layers.ClipBoxes(name='clipped_boxes1')([model1.inputs[0], boxes1])
    boxes2 = layers.RegressBoxes(name='boxes2')([anchors2, regression2])
    boxes2 = layers.ClipBoxes(name='clipped_boxes2')([model2.inputs[0], boxes2])

    # Concatenating all the obtained results

    """boxes = keras.layers.Concatenate(axis=1)([boxes1, boxes2])
    classification = keras.layers.Concatenate(axis=1)([classification1, classification2])
    other = other1 + other2"""

    # filter detections (apply NMS / score threshold / select top-k)
    """detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        max_detections=600,
        nms_threshold=0.5,
        score_threshold=0.05,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)"""

    detections1 = layers.FilterDetectionsFusion(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        max_detections=300,
        nms_threshold=0.486,
        score_threshold=0.05,
        soft_nms_sigma=0.4,
        name                  = 'filtered_detections1'
    )([boxes1, classification1] + other1)

    detections2 = layers.FilterDetectionsFusion(
        nms=nms,
        class_specific_filter=class_specific_filter,
        max_detections=300,
        nms_threshold=0.34,
        score_threshold=0.05,
        soft_nms_sigma=2.0,
        name='filtered_detections2'
    )([boxes2, classification2] + other2)

    boxes_first_nms = keras.layers.Concatenate(axis=1)([detections1[0], detections2[0]])
    scores_first_nms = keras.layers.Concatenate(axis=1)([detections1[1], detections2[1]])
    labels_first_nms = keras.layers.Concatenate(axis=1)([detections1[2], detections2[2]])
    classification_first_nms = keras.layers.Concatenate(axis=1)([detections1[3], detections2[3]])
    other_first_nms = detections1[4:] + detections2[4:]

    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        max_detections=600,
        nms_threshold=0.45,
        score_threshold=0.05,
        name                  = 'filtered_detections'
    )([boxes_first_nms, classification_first_nms] + other_first_nms)

    outputs = detections

    # construct the model

    return keras.models.Model(inputs=[model1.inputs[0], model2.inputs[0]], outputs=outputs, name=name)

def retinanet_bbox_or_fusion(
    model1                = None,
    model2                = None,
    anchor_parameters     = AnchorParameters.default,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    **kwargs
):
    """ Construct two RetinaNet models on top of a backbone, one for each modality and adds convenience functions to output boxes directly.

    This model uses two RetinaNet models, each one specialized on a modality. Predictions are made for each model and the obtained results are
    fused before being filtered in a final step.

    Args
        model1                : RetinaNet model from first modality to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        model2                : RetinaNet model from second modality to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters     : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model1 is None:
        model1 = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    if model2 is None:
        model2 = retinanet(num_anchors=anchor_parameters.num_anchors(),
                           create_pyramid_features=__create_pyramid_features2,
                           regression_name='regression_submodel2',
                           classification_name='classification_submodel2',
                           reg_name='regression2',
                           class_name='classification2',
                           name='retinanet2',
                           **kwargs)

    """if model2 is None:
        model2 = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)"""

    # compute the anchors for both modalities
    features1 = [model1.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors1  = __build_anchors(anchor_parameters, features1)
    features2 = [model2.get_layer(p_name).output for p_name in ['P32', 'P42', 'P52', 'P62', 'P72']]
    anchors2 = __build_anchors2(anchor_parameters, features2)

    # we expect the anchors, regression and classification values as first output for both modalities
    regression1     = model1.outputs[0]
    classification1 = model1.outputs[1]
    regression2 = model2.outputs[0]
    classification2 = model2.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other1 = model1.outputs[2:]
    other2 = model2.outputs[2:]

    # apply predicted regression to anchors for each modality
    boxes1 = layers.RegressBoxes(name='boxes1')([anchors1, regression1])
    boxes1 = layers.ClipBoxes(name='clipped_boxes1')([model1.inputs[0], boxes1])
    boxes2 = layers.RegressBoxes(name='boxes2')([anchors2, regression2])
    boxes2 = layers.ClipBoxes(name='clipped_boxes2')([model2.inputs[0], boxes2])

    # filter detections (apply NMS / score threshold / select top-k)

    detections1 = layers.FilterDetectionsFusion(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        max_detections=300,
        nms_threshold=0.486,
        score_threshold=0.05,
        soft_nms_sigma=0.4,
        name                  = 'filtered_detections1'
    )([boxes1, classification1] + other1)

    detections2 = layers.FilterDetectionsFusion(
        nms=nms,
        class_specific_filter=class_specific_filter,
        max_detections=300,
        nms_threshold=0.34,
        score_threshold=0.05,
        soft_nms_sigma=2.0,
        name='filtered_detections2'
    )([boxes2, classification2] + other2)

    detections = [detections1[0], detections2[0], detections1[1], detections2[1], detections1[2], detections2[2]]

    outputs = detections

    # construct the model

    return keras.models.Model(inputs=[model1.inputs[0], model2.inputs[0]], outputs=outputs, name=name)