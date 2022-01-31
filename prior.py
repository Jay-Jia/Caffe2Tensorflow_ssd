import numpy as np
import tensorflow as tf

class PriorBox(object):
    """Generate the prior boxes of designated sizes and aspect ratios.
    # Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)
    """
    def __init__(self, img_size, fixed_size, density, step, offset=0.5, aspect_ratios=None,
                 flip=False, variances=[0.1], clip=True, **kwargs):
        
        self.waxis = 2
        self.haxis = 1
        self.step = step
        self.offset = offset
        self.img_size = img_size
        self.fixed_size = fixed_size
        self.density = density
        self.aspect_ratios = aspect_ratios
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = False
        super(PriorBox, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        num_priors_ = len(self.aspect_ratios) * len(self.fixed_size)
        for i in self.density:
            num_priors += (pow(i, 2) - 1)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def output(self, x, mask=None):
        input_shape = x.get_shape().as_list()
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[2]
        img_height = self.img_size[1]
        output_data = []
        for h in range(layer_height):
            for w in range(layer_width):
                center_x = (w+self.offset) * self.step
                center_y = (h+self.offset) * self.step
                for s in range(len(self.fixed_size)):
                    box_width = box_height = self.fixed_size[s]
                    dens = int(self.density[s])
                    shift = int(self.fixed_size[s] / dens)
                    for r in range(dens):
                        for c in range(dens):
                            center_x_temp = center_x - self.fixed_size[s] / 2 + shift / 2. + c * shift
                            center_y_temp = center_y - self.fixed_size[s] / 2 + shift / 2. + r * shift
                            center_xmin_temp = (center_x_temp - box_width / 2.) / img_width if (center_x_temp - box_width / 2.) / img_width >=0 else 0
                            center_ymin_temp = (center_y_temp - box_height / 2.) / img_height if (center_y_temp - box_height / 2.) / img_height >=0 else 0
                            center_xmax_temp = (center_x_temp + box_width / 2.) / img_width if (center_x_temp + box_width / 2.) / img_width <=1 else 1
                            center_ymax_temp = (center_y_temp + box_height / 2.) / img_height if (center_y_temp + box_height / 2.) / img_height <=1 else 1
                            output_data.append([center_xmin_temp, center_ymin_temp, center_xmax_temp,  center_ymax_temp])
        output = output_data
        return output

def priorBox(input_data, feature, density, fixed_size, step, offset=0.5):
    input_shape = feature.get_shape().as_list()
    img_size = input_data.get_shape().as_list()
    layer_width = input_shape[2]
    layer_height = input_shape[1]
    img_width = img_size[2]
    img_height = img_size[1]
    output_data = []
    for h in range(layer_height):
        for w in range(layer_width):
            center_x = (w+offset) * step
            center_y = (h+offset) * step
            for s in range(len(fixed_size)):
                box_width = box_height = fixed_size[s]
                dens = int(density[s])
                shift = int(fixed_size[s] / dens)
                for r in range(dens):
                    for c in range(dens):
                        center_x_temp = center_x - fixed_size[s] / 2 + shift / 2. + c * shift
                        center_y_temp = center_y - fixed_size[s] / 2 + shift / 2. + r * shift
                        center_xmin_temp = (center_x_temp - box_width / 2.) / img_width if (center_x_temp - box_width / 2.) / img_width >=0 else 0
                        center_ymin_temp = (center_y_temp - box_height / 2.) / img_height if (center_y_temp - box_height / 2.) / img_height >=0 else 0
                        center_xmax_temp = (center_x_temp + box_width / 2.) / img_width if (center_x_temp + box_width / 2.) / img_width <=1 else 1
                        center_ymax_temp = (center_y_temp + box_height / 2.) / img_height if (center_y_temp + box_height / 2.) / img_height <=1 else 1
                        output_data.append([center_xmin_temp, center_ymin_temp, center_xmax_temp,  center_ymax_temp])
    output_data = tf.convert_to_tensor(output_data)
    output_data = tf.reshape(output_data, tf.stack([p_shape[0], p_shape[-1]]))
    return output_data
