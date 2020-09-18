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

from ..preprocessing.generator import Generator_multi
from ..utils.image import read_image_bgr, read_image_fusion

import os
import numpy as np
from six import raise_from
from PIL import Image

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

voc_classes = {
    'person' : 0,
    'bike': 1,
    'car' : 2,
    'motorbike' : 3
} # PolarLITIS Classes

def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result

class PascalVocGeneratorLF(Generator_multi):
    """ Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
        self,
        data_dir, #/home/rblin/Documents/BD_QCAV
        train_dir_1,
        train_dir_2,
        labels_dir_1,
        #labels_dir_2,
        set_name,
        classes=voc_classes,
        image_extension='.png',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        """ Initialize a Pascal VOC data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """
        self.data_dir             = data_dir
        self.train_dir_1        = train_dir_1
        self.train_dir_2      = train_dir_2
        self.labels_dir_1           = labels_dir_1
        #self.labels_dir_2         = labels_dir_2
        self.set_name             = set_name
        self.classes              = classes
        self.image_names_1 = sorted(os.listdir(os.path.join(self.data_dir, self.train_dir_1)))
        self.image_names_2 = sorted(os.listdir(os.path.join(self.data_dir, self.train_dir_2)))
        self.annotation_names_1     = sorted(os.listdir(os.path.join(self.data_dir, self.labels_dir_1)))
        #self.annotation_names_2     = sorted(os.listdir(os.path.join(self.data_dir, self.labels_dir_2)))
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGeneratorLF, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names_1)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path = os.path.join(self.data_dir, self.train_dir_1, self.image_names_1[image_index])
        image = read_image_bgr(path)
        return float(image.shape[0]) / float(image.shape[1])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path1 = os.path.join(self.data_dir, self.train_dir_1, self.image_names_1[image_index])
        path2 = os.path.join(self.data_dir, self.train_dir_2, self.image_names_2[image_index])
        return [read_image_fusion(path1), read_image_fusion(path2)]

    def __parse_annotation(self, element):
        """ Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((1, 5))
        box[0, 4] = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box

    def __parse_annotations(self, xml_root):
        """ Parse all annotations under the xml_root.
        """
        boxes = np.zeros((0, 5))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue
            boxes = np.append(boxes, box, axis=0)

        return boxes

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """

        filename1 = self.annotation_names_1[image_index]
        try:
            tree1 = ET.parse(os.path.join(self.data_dir, self.labels_dir_1, filename1))
            return self.__parse_annotations(tree1.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename1, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename1, e)), None)
