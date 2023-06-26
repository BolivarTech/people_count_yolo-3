"""
   YOLO V3 OpenVINO
"""

import os
import sys
import logging
from math import exp as exp
import random as rand
# Drawing imports
import cv2
import colorsys

from openvino.inference_engine import IENetwork, IEPlugin, IECore

from YoloParams import YoloParams

log = logging.getLogger()

class YOLO:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, obj_threshold=0.6, box_threshold=0.5,classesPath='models/coco_classes.txt'):
        self.__net = None
        self.__plugin = None
        self.__input_blob = None
        self.__out_blob = None
        self.__net_plugin = None
        self.__infer_request_handle = None
        # Detection thresholds
        self.__objT = obj_threshold
        self.__boxT = box_threshold
        # YOLO Classes
        self.__class_names = self.__get_classes(classesPath)
        self.__classColors = self.__generate_colors(self.__class_names)

    def __del__(self):
        """
            Class Destructor
            Deletes all the instances

            :return: None
        """
        del self.__net_plugin
        del self.__plugin
        del self.__net
        del self.__infer_request_handle

    def __get_classes(self, file):
        """
            Get classes name.

            :param file: classes name for database.
            :returns class_names: List, classes name.
        """
        try:
            with open(file, 'r') as f:
                class_names = f.readlines()
        except OSError as fError:
            class_names = None
            log.error(f"File ERROR: {fError}  (003)")
        else:
            class_names = [c.strip() for c in class_names]
        finally:
            return class_names

    def __generate_colors(self, class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        rand.seed(10101)  # Fixed seed for consistent colors across runs.
        rand.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        rand.seed(None)  # Reset seed to default.
        return colors

    def load_model(self, model='models/yolov3_model.xml', device='CPU', input_size=1, output_size=3, num_requests=0, cpu_extension=None, plugin=None):
        """
         Loads a network and an image to the Inference Engine __plugin.
        :param model: .xml file of pre trained model
        :param cpu_extension: extension for the CPU device
        :param device: Target device
        :param input_size: Number of input layers
        :param output_size: Number of output layers
        :param num_requests: Index of Infer request value. Limited to device capabilities.
        :param plugin: Plugin for specified device
        :return:  Shape of input layer
        """

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        ie = IECore()

        # Plugin initialization for specified device
        # and load extensions library if specified
        if not plugin:
            log.info("Initializing __plugin for {} device...".format(device))
            ie.register_plugin(plugin_name=device)
            self.__plugin = IEPlugin(device=device)
        else:
            self.__plugin = plugin

        if cpu_extension and 'CPU' in device:
            # ie.add_extension(extension_path=cpu_extension, device_name="CPU")
            self.__plugin.add_cpu_extension(cpu_extension)

        # Read IR
        log.info("Reading IR...")
        self.__net = ie.read_network(model=model_xml, weights=model_bin)
        # self.__net = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR to the __plugin...")

        if self.__plugin.device == "CPU":
            supported_layers = self.__plugin.get_supported_layers(self.__net)
            not_supported_layers = \
                [l for l in self.__net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by "
                          "the __plugin for specified device {}:\n {}".
                          format(self.__plugin.device,
                                 ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        if num_requests == 0:
            # Loads network read from IR to the __plugin
            self.__net_plugin = self.__plugin.load(network=self.__net)
        else:
            self.__net_plugin = self.__plugin.load(network=self.__net, num_requests=num_requests)

        self.__input_blob = next(iter(self.__net.inputs))
        self.__out_blob = next(iter(self.__net.outputs))
        assert len(self.__net.inputs.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.__net.inputs))
        assert len(self.__net.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.__net.outputs))

        return self.__plugin, self.get_input_shape()

    def get_input_shape(self):
        """
        Gives the shape of the input layer of the network.
        :return: None
        """
        return self.__net.inputs[self.__input_blob].shape

    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer  
        """
        perf_count = self.__net_plugin.requests[request_id].get_perf_counts()
        return perf_count

    def infer(self, request_id, frame):
        """
        Starts asynchronous inference for specified request.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param frame: Input image
        :return: Instance of Executable YOLO class
        """
        self.__infer_request_handle = self.__net_plugin.start_async(
            request_id=request_id, inputs={self.__input_blob: frame})
        return self.__net_plugin

    def wait(self, request_id):
        """
        Waits for the result to become available.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :return: Timeout value
        """
        wait_process = self.__net_plugin.requests[request_id].wait(-1)
        return wait_process

    def get_output(self, request_id, output=None):
        """
        Gives a list of results for the output layer of the network.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param output: Name of the output layer
        :return: Results for the specified request
        """
        if output:
            res = self.__infer_request_handle.outputs[output]
        else:
            res = self.__net_plugin.requests[request_id].outputs
        return res

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.__net_plugin
        del self.__plugin
        del self.__net

    def __parse_yolo_region(self, blob, resized_image_shape, original_im_shape, params, threshold):
        """

        :param blob:
        :param resized_image_shape:
        :param original_im_shape:
        :param params:
        :param threshold:
        :return:
        """
        # ------------------------------------------ Validating output parameters ------------------------------------------
        _, _, out_blob_h, out_blob_w = blob.shape
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                         "be equal to width. Current height = {}, current width = {}" \
                                         "".format(out_blob_h, out_blob_w)

        # ------------------------------------------ Extracting layer parameters -------------------------------------------
        orig_im_h, orig_im_w = original_im_shape
        resized_image_h, resized_image_w = resized_image_shape
        objects = list()
        predictions = blob.flatten()
        side_square = params.side * params.side

        # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
        for i in range(side_square):
            row = i // params.side
            col = i % params.side
            for n in range(params.num):
                obj_index = self.__entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
                scale = predictions[obj_index]
                if scale < threshold:
                    continue
                box_index = self.__entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
                # YOLO produces location predictions in absolute coordinates of feature maps.
                # Scale it to relative coordinates.
                x = (col + predictions[box_index + 0 * side_square]) / params.side
                y = (row + predictions[box_index + 1 * side_square]) / params.side
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    w_exp = exp(predictions[box_index + 2 * side_square])
                    h_exp = exp(predictions[box_index + 3 * side_square])
                except OverflowError:
                    continue
                # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
                w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
                h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
                for j in range(params.classes):
                    class_index = self.__entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                                     params.coords + 1 + j)
                    confidence = scale * predictions[class_index]
                    if confidence < threshold:
                        continue
                    objects.append(self.__scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                                     h_scale=orig_im_h, w_scale=orig_im_w))
        return objects

    def __intersection_over_union(self, box_1, box_2):
        """

        :param box_1:
        :param box_2:
        :return:
        """

        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union

    def image_preprocess(self, frame):
        """
        Given one frame preprocess it and return a frame YOLO shape.
        :param frame: Original frame.
        :return: frame resized for YOLOv3
        """

        n, c, h, w = self.get_input_shape()
        self.org_frame = frame
        self.in_frame = cv2.resize(self.org_frame, (w, h))
        # resize input_frame to network size
        self.in_frame = self.in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.in_frame = self.in_frame.reshape((n, c, h, w))
        return self.in_frame

    def yolo_out(self, output):
        """Process output of yolo base __net.

        # Argument:
            output: output of yolo base __net.
        # Returns:
            objets: list of object's metadata.
            labels: list of classes of objects.
        """

        objects = list()
        labels = list()
        det_objets = list()

        for layer_name, out_blob in output.items():
            out_blob = out_blob.reshape(self.__net.layers[self.__net.layers[layer_name].parents[0]].shape)
            layer_params = YoloParams(self.__net.layers[layer_name].params, out_blob.shape[2])
            log.info("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += self.__parse_yolo_region(out_blob, self.in_frame.shape[2:],
                                              self.org_frame.shape[:-1], layer_params,
                                                self.__objT)
            # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
            objects = sorted(objects, key=lambda obj: obj['confidence'], reverse=True)
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if self.__intersection_over_union(objects[i], objects[j]) > self.__boxT:
                        objects[j]['confidence'] = 0

            # Drawing objects with respect to the --prob_threshold CLI parameter
            objects = [obj for obj in objects if obj['confidence'] >= self.__objT]

            if len(objects):
                log.info("\nDetected boxes for batch {}:".format(1))
                log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

            origin_im_size = self.org_frame.shape[:-1]
            for obj in objects:
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                    continue
                det_label = self.__class_names[obj['class_id']] if self.__class_names and len(self.__class_names) >= \
                                                                   obj['class_id'] else \
                    str(obj['class_id'])
                det_objets.append(obj)
                labels.append(det_label)
                log.info(
                    "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} ".format(det_label, obj['confidence'],
                                                                         obj['xmin'],
                                                                         obj['ymin'], obj['xmax'],
                                                                         obj['ymax']))
        return det_objets, labels

    def draw_boxes(self, frame, det_objects, labels):

        origin_im_size = frame.shape[:-1]
        for obj,det_label in zip(det_objects,labels):
            # Validation bbox of detected object
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue
            color = self.__classColors[obj['class_id']]
            log.info("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                              obj['ymin'], obj['xmax'], obj['ymax']))

            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,
                        "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                        (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        return frame

    def __entry_index(self, side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


    def __scale_bbox(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        xmin = int((x - w / 2) * w_scale)
        ymin = int((y - h / 2) * h_scale)
        xmax = int(xmin + w * w_scale)
        ymax = int(ymin + h * h_scale)
        return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)