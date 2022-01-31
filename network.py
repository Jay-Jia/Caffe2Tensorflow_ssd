import caffe_pb2 as cpb
from google.protobuf import text_format
from collections import defaultdict
import tensorflow as tf
import numpy as np
from operator import mul
import logging
# import .detection as dtc
from .detection import tf_ssd_bboxes_select_layer, tf_ssd_bboxes_decode_layer
from .prior import PriorBox


class Network:
    # Public functions.
    def __init__(self, proto, inputs=dict(), input_format="NC*"):
        """Create network using structure from Caffe proto file.

        Args:
            proto: The plaintext network prototxt file on disk.
                Type `str`.
            inputs: Dictionary mapping input variable `top` strings to
                Tensors. Use this to connect Tensors directly to the
                network, instead of using `tf.placeholder` type variables
                that are created by default for all layers of type
                `Input`. `tf.placeholder` instances will be created for
                any `top` values missing in the dictionary. Defaults to
                empty `dict`.
            input_format: The order of inputs specified in the prototxt
                file. The TensorFlow program is expected to provide inputs
                in the order `N*C`, where the dimension `N` denotes batch,
                `C` denotes the number of channels, and `*` denotes the
                rest of the dimensions. Valid arguments are `N*C`, `NC*`,
                `*NC`, `CN*`, `C*N`, and `*CN`. The network will permute
                the order of dimensions specified in prototxt file for the
                InputParameter in the order `N*C`. Defaults to `NC*`.
        """
        # Layer names mapped to outputs.
        self._name_outputs = defaultdict(list)
        # Layer top names mapped to outputs.
        self._top_outputs = defaultdict(list)
        # Layer name to list of variables.
        self._name_vars = defaultdict(list)
        # Protobuf LayerParameter messages for each layer, loaded in the
        # order presented in the proto file.
        self._layer_params = []
        # List of variables.
        self._vars = []
        # List of trainable variables.
        self._tvars = []
        # Layer name to (name, var) mapping.
        self._layer_vars = defaultdict(list)
        # The NetParameter protobuf message.
        self._net_param = cpb.NetParameter()
        self._input_format = input_format
        self._inputs = dict(inputs)
        

        # Load and parse the prototxt file.
        self._parse_proto(proto)
        # print(self._parse_proto)
        # Create network, based on the parse.
        self._make_network()

    def load(self, model, sess):
        """Load network variables from NumPy file on disk.

        Args:
            model: The path to the model file on disk.
            sess: The TensorFlow session.
        """
        self._load(model, sess)

    def layer_output(self, layer_name):
        if layer_name not in self._top_outputs:
            raise ValueError("`%s` not defined in network" % layer_name)
        return self._top(layer_name)

    def save(self, path, sess):
        """Save the network weights to disk.

        Saved as compatible NumPy pickle.

        Args:
            path: The path of save file on disk.
            sess: The TensorFlow session.
        """
        m = {}
        for layer_name, name_var_list in self._layer_vars.iteritems():
            m[layer_name] = {}
            for var_name, var in name_var_list:
                m[layer_name][var_name] = var.eval(sess)
        np.save(path, m)

    @property
    def vars(self):
        """The network variables.

        Returns:
            List of all network TensorFlow variables.
        """
        return self._vars

    @property
    def tvars(self):
        """Trainable network variables.

        Returns:
            List of all trainable network TensorFlow variables.
        """
        return self._tvars

    # Layer functions.
    def _add_Input(self, lp):
        """Add input layer placeholder to network.

        Does not support undefined shapes.

        Args:
            lp: Caffe LayerParameter protobuf message
                with type "Input".
        """
        def reorder(dims):
            # Figure out the permutation for the shape to convert to N*C order.
            idx_C = self._input_format.index("C")
            if idx_C == 2:
                idx_C = -1
            idx_N = self._input_format.index("N")
            if idx_N == 2:
                idx_N = -1
            # Reorder.
            new_dim = [dims[idx_N]]
            temp = list(dims)
            temp = [i for j, i in enumerate(dims) if j not in [idx_C, idx_N]]
            new_dim += (temp + [dims[idx_C]])
            return new_dim

        # In case of multiple outputs but one shape specification, repeat.
        if len(lp.top) != len(lp.input_param.shape):
            shapes = lp.input_param.shape * len(lp.top)
        else:
            shapes = lp.input_param.shape
        for (top, shape) in zip(lp.top, shapes):
            # Reorder shape to get `N*C` ordering.
            shape_reordered = tuple(map(int, reorder(shape.dim)))
            # Check if inputs have been supplied by user.
            if top in self._inputs:
                input = self._inputs[top]
                # print(input)
                # Make sure user-specified Tensor shape is compatible
                # with proto shape specifications.
                if not input.get_shape().is_compatible_with(shape_reordered):
                    raise ValueError("Shape of supplied `%s` input incompatible"
                                     " with proto specifications" % top)
                output = input
               
            else:
                output = tf.placeholder(tf.float32, shape_reordered, name=top)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Convolution(self, lp):
        """Add convolution layer to network.

        Note that only 2-D convolutions are currently supported.

        Args:
            lp: Caffe LayerParameter protobuf message
                with type "Convolution".
        """
        filter_size, padding, strides, out_channels = \
            self._parse_ConvolutionParameter(lp.convolution_param)
        # Get the number of chanels in input. Ensure that the number
        # is consistent for each of the bottom layers.
        in_channels = 0
        
        for bottom in lp.bottom:
            if in_channels == 0:
                # print(self._top(bottom))
                in_channels = self._top(bottom).get_shape()[-1]
            if in_channels != self._top(bottom).get_shape()[-1]:
                raise ValueError("bottom layers for %s must have same "
                                 "channel count" % lp.name)
        # Create convolution filter weights.
        with tf.variable_scope(lp.name):
            weights = self._make_vars(
                "weights",
                filter_size + [in_channels, out_channels],
                self._is_training(lp.phase),
                lp.name,
                self._FillerParameter_to_initializer(
                    lp.convolution_param.weight_filler, self._is_training(lp.phase)))
            if lp.convolution_param.bias_term:
                biases = self._make_vars(
                    "biases",
                    [out_channels],
                    self._is_training(lp.phase),
                    lp.name,
                    self._FillerParameter_to_initializer(
                        lp.convolution_param.bias_filler, self._is_training(lp.phase)))
        # For each bottom layer, create convolutional output
        # feature map.
        for (top, bottom) in zip(lp.top, lp.bottom):
            input = self._top(bottom)
            # Perform 2D convolution.
            with tf.name_scope(self._name_scope_builder(top)):
                # Pad input, if necessary.
                padded_input = tf.pad(input, padding)
                # Convolve.
                output = tf.nn.conv2d(padded_input, weights, strides, "VALID")
                # Add biases, if needed.
                if lp.convolution_param.bias_term:
                    output = tf.nn.bias_add(output, biases)
            # Add to lists.
            # print(biases)
            self._add_output_to_lists(output, lp.name, top)

    def _add_ReLU(self, lp):
        if len(lp.top) != len(lp.bottom):
            raise ValueError(
                "ReLU layers (%s) must have the same number of "
                "top and bottom layers" % lp.name)
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.nn.relu(self._top(bottom), name=lp.name)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Pooling(self, lp):
        if len(lp.top) != len(lp.bottom):
            raise ValueError(
                "Pooling layers (%s) must have the same number of "
                "top and bottom layers" % lp.name)

        for top, bottom in zip(lp.top, lp.bottom):
            pp = lp.pooling_param
            # Recover the pooling parameters.
            input = self._top(bottom)
            ksize, padding, strides = self._parse_PoolingParameter(
                pp, input)
            with tf.name_scope(self._name_scope_builder(lp.name)):
                # Pad, if necessary.
                if padding is not None:
                    input = tf.pad(input, padding)
                if pp.pool == cpb.PoolingParameter.MAX:
                    if pp.stride == 2:
                        output = tf.nn.max_pool(input, ksize, strides, "SAME")
                    else:
                        output = tf.nn.max_pool(input, ksize, strides, "VALID")
                elif pp.pool == cpb.PoolingParameter.AVE:
                    output = tf.nn.avg_pool(input, ksize, strides, "VALID")
                else:
                    raise NotImplementedError(
                        "Unsupported pooling: %s" % pp.pool)
            self._add_output_to_lists(output, lp.name, top)

    def _add_InnerProduct(self, lp):
        # TODO(agharwal): Add support for multiple top and bottom layers.
        #                 Users should use Concat and Reshape layers to
        #                 concatenate multiple bottom layers.
        if len(lp.top) > 1:
            raise ValueError(
                "InnerProduct layers (%s) must have exactly one "
                "top layer" % lp.name)

        inputs = []
        with tf.name_scope(self._name_scope_builder(lp.name)):
            for bottom in lp.bottom:
                input = self._top(bottom)
                shape = input.get_shape().as_list()
                # Get the flattened shape, excluding the index
                # correspnding to batch.
                ndims_flat = reduce(mul, shape[1:], 1)  # NOQA
                inputs.append(tf.reshape(input, [shape[0], ndims_flat]))
            # Concatenate flattened inputs.
            if len(inputs) > 1:
                input = tf.concat(1, inputs)
            else:
                input = inputs[0]

        # Prepare weights and biases.
        ipp = lp.inner_product_param
        input_ndims = input.get_shape().as_list()[-1]
        with tf.variable_scope(lp.name):
            weights = self._make_vars(
                "weights",
                [input_ndims, ipp.num_output],
                self._is_training(lp.phase),
                lp.name,
                self._FillerParameter_to_initializer(
                    ipp.weight_filler, self._is_training(lp.phase)))
            if ipp.bias_term:
                biases = self._make_vars(
                    "biases",
                    [ipp.num_output],
                    self._is_training(lp.phase),
                    lp.name,
                    self._FillerParameter_to_initializer(
                        ipp.bias_filler, self._is_training(lp.phase)))

        # Compute output.
        with tf.name_scope(self._name_scope_builder(lp.name)):
            if ipp.bias_term:
                output = tf.nn.xw_plus_b(input, weights, biases)
            else:
                output = input * weights
        self._add_output_to_lists(output, lp.name, lp.top[0])

    def _add_Dropout(self, lp):
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.nn.dropout(self._top(bottom),
                                       1.0 - lp.dropout_param.dropout_ratio,
                                       name=lp.name)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Softmax(self, lp):
        if len(lp.top) != len(lp.bottom):
            raise ValueError(
                "Softmax layers (%s) must have the same number of "
                "top and bottom layers" % lp.name)
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.nn.softmax(self._top(bottom), axis = lp.softmax_param.axis)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Flatten(self, lp):
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.layers.flatten(self._top(bottom),name=lp.name)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Permute(self, lp):
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = self._top(bottom)
            self._add_output_to_lists(output, lp.name, top)
        # self._add_output_to_lists(self._top(lp.bottom), lp.name, top)

    def _add_Reshape(self, lp):
        # FIXME(agharwal): Possible bug since this is not compensating
        #                  for order of axes in input.
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.reshape(
                    self._top(bottom),
                    self._ReshapeParameter_to_shape(
                        lp.reshape_param,
                        self._top(bottom).get_shape().as_list()),
                    name=lp.name)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Sigmoid(self, lp):
        if len(lp.top) != len(lp.bottom):
            raise ValueError(
                "Sigmoid layers (%s) must have the same number of "
                "top and bottom layers" % lp.name)
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.sigmoid(self._top(bottom), name=lp.name)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Concat(self, lp):
        if len(lp.top) != 1:
            raise ValueError(
                "Concat layers (%s) must have exactly one "
                "top layer" % lp.name)
        axis = self._parse_ConcatParameter(lp.concat_param)
        if axis == 1:
            axis =3
        elif axis>=2:
            axis = axis -1
        top = lp.top[0]
        if lp.name == "mbox_priorbox":
            axis = 0
        if lp.name == "mbox_loc" or lp.name == "mbox_conf":
            axis = 1
        bottom_tensors = [self._top(name) for name in lp.bottom]
        with tf.name_scope(self._name_scope_builder(top)):
            output = tf.concat(bottom_tensors, axis, name=lp.name)
        self._add_output_to_lists(output, lp.name, top)

    def _add_Prior(self, lp):
        if len(lp.top) != 1:
            raise ValueError(
                "Prior options (%s) must have exactly one "
                "top layer" % lp.name)
        fixed_size = lp.prior_box_param.fixed_size
        density = lp.prior_box_param.density
        step = lp.prior_box_param.step
        offset = lp.prior_box_param.offset
        bottom_tensors = [self._top(name) for name in lp.bottom]
        top = lp.top[0]
        with tf.name_scope(self._name_scope_builder(top), 'ssd_bboxes_prior_layer', [bottom_tensors[0], bottom_tensors[1]]):
            img_size = bottom_tensors[1].get_shape().as_list()
            output_data = []
            input_shape = bottom_tensors[0].get_shape().as_list()
            layer_width = input_shape[2]
            layer_height = input_shape[1]
            img_width = img_size[2]
            img_height = img_size[1]
            for h in range(layer_height):
                for w in range(layer_width):
                    center_x = (w+ offset) * step
                    center_y = (h+ offset) * step
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
        self._add_output_to_lists(output_data, lp.name, top)
        
    def _add_Detection(self, lp):
        select_threshold = lp.detection_output_param.nms_param.nms_threshold
        num_classes = lp.detection_output_param.num_classes
        top_k = lp.detection_output_param.nms_param.top_k
        loc = self._top(lp.bottom[0])
        conf = self._top(lp.bottom[1])
        prior = self._top(lp.bottom[2])
        top = lp.top[0]
        loc = tf.reshape(loc, shape=[-1, 4])
        keep_top_k = lp.detection_output_param.keep_top_k
        confidence_threshold = lp.detection_output_param.confidence_threshold
        background_label_id = lp.detection_output_param.background_label_id
        output = []
        with tf.name_scope(self._name_scope_builder(top)):
            loc1 = tf_ssd_bboxes_decode_layer(loc, prior)
            a,b = tf_ssd_bboxes_select_layer(conf, loc1, confidence_threshold, num_classes, top_k)
            output = [a,b]
        self._add_output_to_lists(output, lp.name, top)
        # print(decode_box)

    def _add_Slice(self, lp):
        if len(lp.bottom) != 1:
            raise ValueError(
                "Concat layers (%s) must have exactly one "
                "top layer" % lp.name)
        bottom_tensor = self._top(lp.bottom[0])
        axis, slice_points = self._parse_SliceParameter(lp.slice_param)
        # Recover the total number of axes, and the number of
        # dimensions along `axis`.
        bottom_shape = bottom_tensor.get_shape().as_list()
        num_axes = len(bottom_shape)
        top_layer_names = lp.top
        # Ensure that the number of slices match the number of top layers.
        if len(top_layer_names) != (len(slice_points) + 1):
            raise ValueError(
                "Slice layers (%s) must have exactly one more"
                "top layer than slice points" % lp.name)
        # Make sure the axis index is within range. Note that we need to take
        # into account negative indices too.
        if axis < -num_axes or axis >= num_axes:
            raise ValueError(
                "Slice layer (%s) axis index mismatch %d for %d axes" %
                (lp.name, axis, num_axes))
        start_indices = [0] + slice_points
        end_indices = slice_points + [bottom_shape[axis]]
        for (start_idx, end_idx, top) in zip(start_indices,
                                             end_indices,
                                             top_layer_names):
            begin = [0] * num_axes
            size = bottom_shape
            begin[axis] = start_idx
            size[axis] = end_idx - start_idx
            with tf.name_scope(self._name_scope_builder(top)):
                output = tf.slice(bottom_tensor, begin, size, name=lp.name)
                self._add_output_to_lists(output, lp.name, top)

    # Helper functions.
    def _make_vars(self, name, shape, trainable,
                   layer_name=None, initializer=None):
        """Create variables of given `name` and `shape`.

        Args:
            name: Variable name. Type `str`.
            shape: A 1-D `Tensor` of type `int32` representing
                the shape of the variables.
            trainable: `True` if variables are trainable.
            layer_name: Name of layer for which variables will be used.
                Type `str`. Ignored if set to `None`. Defaults to `None`.
        """
        vars = tf.get_variable(name, shape,
                               trainable=trainable,
                               initializer=initializer)
        self._vars.append(vars)
        if trainable:
            self._tvars.append(vars)
        if layer_name is not None:
            self._layer_vars[layer_name].append((name, vars))
        return vars

    def _add_output_to_lists(self, output, name, top):
        self._top_outputs[top].append(output)
        self._name_outputs[name].append(
            (top, len(self._top_outputs[top]) - 1, output))

    def _parse_proto(self, proto):
        """Parse the proto file to create network graph

        Args:
            proto: Path of prototxt plaintext file on disk.
                Type `str`.
        """
        with open(proto, "r") as f:
            text_format.Merge(f.read(), self._net_param)
            for layer_param in self._net_param.layer:
                self._layer_params.append(layer_param)

    def _top(self, top):
        # print(top)
        # print(self._top_outputs[top])
        return self._top_outputs[top][-1]

    def _make_network(self):
        """Construct the network using the parsed proto file."""
        for layer_param in self._layer_params:
            # Does layer need to be skipped?
            # print(layer_param)
            if not self._include_layer(layer_param):
                logging.info("Skipping: %s", layer_param.name)
                continue
            if layer_param.type == "Input":
                self._add_Input(layer_param)
            elif layer_param.type == "Convolution":
                self._add_Convolution(layer_param)
            elif layer_param.type == "ReLU":
                self._add_ReLU(layer_param)
            elif layer_param.type == "Pooling":
                self._add_Pooling(layer_param)
            elif layer_param.type == "InnerProduct":
                self._add_InnerProduct(layer_param)
            elif layer_param.type == "Dropout":
                self._add_Dropout(layer_param)
            elif layer_param.type == "Softmax":
                self._add_Softmax(layer_param)
            elif layer_param.type == "Reshape":
                self._add_Reshape(layer_param)
            elif layer_param.type == "Sigmoid":
                self._add_Sigmoid(layer_param)
            elif layer_param.type == "Concat":
                self._add_Concat(layer_param)
            elif layer_param.type == "Slice":
                self._add_Slice(layer_param)
            elif layer_param.type == "PriorBox":
                self._add_Prior(layer_param)
            elif layer_param.type == "Permute":
                self._add_Permute(layer_param)
            elif layer_param.type == "Flatten":
                self._add_Flatten(layer_param)
            elif layer_param.type == "DetectionOutput":
                self._add_Detection(layer_param)
            else:
                # print(layer_param.type)
                raise NotImplementedError(
                    "Contact the bugger who wrote this.")

    def _is_training(self, phase):
        """Global phase overrides local phase in case the network is
           in test phase; local phase is used otherwise.
        """
        return self._net_param.state.phase == cpb.TRAIN and phase == cpb.TRAIN

    def _parse_ConvolutionParameter(self, cp, dims=2):
        out_channels = cp.num_output

        # Infer input dimensions.
        if len(cp.kernel_size) == 1:
            filter_size = [cp.kernel_size[0]] * dims
        else:
            filter_size = cp.kernel_size

        # Infer padding width.
        if not cp.pad:
            padding = [0] * dims
        elif len(cp.pad) == 1:
            padding = [cp.pad[0]] * dims
        else:
            padding = cp.pad
        # TensorFlow allows specification of different paddings
        # at both ends of an input axis, but Caffe assumes it would
        # be the same. Create a TF compatible output.
        padding = [[0, 0]] + [[p, p] for p in padding] + [[0, 0]]

        # Infer stride.
        if len(cp.stride) == 0:
            strides = [1] * dims
        elif len(cp.stride) == 1:
            strides = [cp.stride[0]] * dims
        else:
            strides = cp.stride
        strides = [1] + strides + [1]

        return filter_size, padding, strides, out_channels

    def _parse_PoolingParameter(self, pp, bottom):
        # Input Tensor dimensions, ignoring batch size and number of channels.
        ndims = len(bottom.get_shape()) - 2
        # If pooling is global.
        if pp.global_pooling:
            # Kernel size equals to full extent of feature map.
            ksize = bottom.get_shape().as_list()[1:-1]
            # No padding.
            padding = None
            # Fixed stride.
            strides = [1] * (ndims + 2)
            return ksize, padding, strides

        # Figure out kernel size.
        # a) kernel_h and kernel_w are set. Feature dimensions
        #    must be equal to 2.
        if pp.kernel_size is None:
            if ndims != 2:
                raise ValueError(
                    "Either kernel_size or both kernel_x and kernel_y"
                    " must be set.")
            ksize = [pp.kernel_h, pp.kernel_w]
        # b) kernel_size is set
        else:
            ksize = [pp.kernel_size] * ndims
        # Adjust for TensorFlow specs. Kernel should be 1 layer
        # wide for batch and channel Tensor indices.
        ksize = [1] + ksize + [1]

        # Figure out stride.
        # a) stride is set
        if pp.stride > 0:
            strides = [pp.stride] * ndims
        # b) stide_h and stride_w are set (we assume both).
        else:
            strides = [pp.stride_h, pp.stride_w]
        # Adjust for TensorFlow specs. Max should be computed
        # over each feature map for each instance in the batch.
        strides = [1] + strides + [1]

        # Infer padding width. No padding along batch and channel
        # indices
        if pp.pad > 0:
            padding = [0] + ([pp.pad] * ndims) + [0]
        elif pp.pad_h > 0 or pp.pad_w > 0:
            padding = [0, pp.pad_h, pp.pad_w, 0]
        else:
            padding = None
        # TensorFlow allows specification of different paddings
        # at both ends of an input axis, but Caffe assumes it would
        # be the same. Create a TF compatible output.
        if padding is not None:
            padding = [[p, p] for p in padding]

        return ksize, padding, strides

    def _parse_ConcatParameter(self, cp):
        # Return the axis index.
        return cp.axis

    def _parse_SliceParameter(self, sp):
        # Return the axis index.
        axis = sp.axis
        slice_points = [slice_point for slice_point in sp.slice_point]
        return axis, slice_points

    # def _parse_ProirBoxParameter(self, pbp, top):
    #     # Return the axis index.
    #     # print(self._top(top[0]))
    #     # ksize = top.get_shape().as_list()[1:-1]

    #     fixed_sizes = pbp.fixed_size
    #     print(fixed_sizes)
    #     # slice_points = [slice_point for slice_point in sp.slice_point]
    #     return fixed_sizes

    def _load(self, model, sess):
        # Load data from file.
        data_dict = np.load(model, allow_pickle=True, encoding='bytes').item()
        # print(data_dict['conv2'])
        # For each layer type, load data.
        for layer_proto in self._layer_params:
            layer_name = layer_proto.name
            fix_layer_name = layer_name.replace('/', '_')

            # Make sure layer has variables.
            if not self._has_vars(layer_proto):
                continue
            # It is possible that the file is missing data
            # for some layers. Warn the user.
            if fix_layer_name not in data_dict:
                logging.warning("No variables found for layer: %s",
                                fix_layer_name)
                continue
            # Load each variable in layer.
            for (name, vars) in self._layer_vars[layer_name]:
                # Still possible for stuff to be missing. This is unusual.
                # Warn the poor soul.
                # print(data_dict[fix_layer_name])
                # print(name.encode('utf-8') in data_dict[fix_layer_name].keys())
                if name.encode('utf-8') not in data_dict[fix_layer_name]:
                    # print("b'weights'" == data_dict[layer_name].keys()[0])
                    logging.warning("Variables %s not found for layer %s",
                                    name,
                                    layer_name)
                    continue
                try:
                    # Time to assign values.
                    name = name.encode('utf-8')
                    # print(data_dict[fix_layer_name][name])
                    sess.run(vars.assign(data_dict[fix_layer_name][name]))
                except ValueError as e:
                    # Something is very wrong. User likely has the wrong
                    # model file.
                    logging.warning("Dimension mismatch for variables"
                                    "%s in layer %s. %s.",
                                    name,
                                    layer_name,
                                    str(e))

    def _has_vars(self, layer_proto):
        return layer_proto.name in self._layer_vars

    def _satisfied_NSR(self, nsr):
        """Check if `NetStateRule` is satisfied based on `NetState`.

        Args:
            nsr: `NetStateRule` protobuf message.

        Returns:
            Boolean `True` iff `NetState` agrees with `nsr`.
        """
        np_state = self._net_param.state
        # Phase check.
        if nsr.phase != np_state.phase:
            return False

        # Level check.
        if np_state.level < nsr.min_level or np_state.level > nsr.max_level:
            return False

        # Stage check.
        stages = set(np_state.stage)
        for stage in nsr.stage:
            # Missing stage? Exclude.
            if stage not in stages:
                return False
        for not_stage in nsr.not_stage:
            # Stage should be there, but found? Exclude.
            if not_stage in stages:
                return False

        return True

    def _include_layer(self, lp):
        """Check if layer should be included in the network.

        Based on `NetState` of the `NetParameter`.

        Args:
            lp: `LayerParameter` protobuf message.

        Returns:
            Boolean `True` iff layer should be included.
        """
        # Default behavior.
        default = True
        # If inclusion rules defined, exclude by default.
        if lp.include:
            default = False
        # If exclusion rules defined, include by default.
        if lp.exclude:
            default = True

        # Include?
        for include_NSR in lp.include:
            if self._satisfied_NSR(include_NSR):
                return True
        # Exclude?
        for exclude_NSR in lp.exclude:
            if self._satisfied_NSR(exclude_NSR):
                return False
        return default

    def _FillerParameter_to_initializer(self, fp, istraining):
        """FillerParameter protobuf message to TensorFlow initializer.

        It is the caller's responsibility to ensure appropriate and
        consistent values are assigned to the FillerParameter fields.

        Args:
            fp: Protobuf message of type `FillerParameter`

        Returns:
            A TensorFlow initializer op.
        """
        if fp.type == "constant":
            return tf.constant_initializer(fp.value)
        elif fp.type == "uniform":
            return tf.random_uniform_initializer(fp.min, fp.max)
        elif fp.type == "gaussian":
            return tf.random_normal_initializer(fp.mean, fp.std)
        # add xavier initializer by zihui
        elif fp.type == "xavier":
            return tf.contrib.layers.xavier_initializer()
        else:
            raise NotImplementedError("Initializer type %s not implemented" %
                                    fp.type)


    def _ReshapeParameter_to_shape(self, rp, bottom_shape):
        """ReshapeParameter protobuf message to shape.

        Args:
            rp: Protobuf message of type `ReshapeParameter`.
            bottom_shape: Shape of bottom layer to reshape. List or
                tuple of `int`.

        Return:
            List of `int` based on `ReshapeParameter` specs.

        Raises:
            `ValueError` in case of `ReshapeParameter` fields are
            incompatible with `bottom_shape`.
        """
        # First, figure out what dimensions of the bottom layer we
        # are working with, leaving behind a prefix and a suffix.
        bottom_len = len(bottom_shape)
        start_idx = (rp.axis + bottom_len) % bottom_len
        end_idx = bottom_len
        if rp.num_axes != -1:
            end_idx = start_idx + rp.num_axes + 1
        working_set = bottom_shape[start_idx:end_idx]

        # Reshape them.
        new_shape = [1] * len(rp.shape.dim)
        unk_idx = -1
        # Forward pass until -1 encountered.
        for idx, dim in enumerate(rp.shape.dim):
            if dim == 0:
                new_shape[idx] = working_set[idx]
            elif dim == -1:
                unk_idx = idx
                # break
            else:
                new_shape[idx] = dim
        prod = 1
        new_prod = 1
        for i in bottom_shape:
            prod = prod * i
        for j in new_shape:
            new_prod = new_prod * j
        new_shape[unk_idx] = prod // new_prod
        # If -1 encountered at `unk_idx`, we need a backward pass to
        # determine all remaining dimensions, except the one at the
        # aforementioned spot.
        # if unk_idx != -1:
        #     # Iterate backwards through `new_shape`, until,
        #     # but excluding the index `unk_idx`.
        #     for idx, dim in enumerate(new_shape[:unk_idx:-1]):
        #         if dim == 0:
        #             new_shape[-idx - 1] = working_set[-idx - 1]
        #         # Definitely not supposed to find a second
        #         # instance of -1.
        #         elif dim == -1:
        #             raise ValueError("More than one instance of -1 encountered")
        #         else:
        #             new_shape[-idx - 1] = dim
        #     # Now, on to figuring out the shape of the
        #     # mystery dimension.
        #     working_set_prod = np.prod(working_set)
        #     new_shape_prod = np.prod(new_shape)
            # Just in case the user messed up.
            # if working_set_prod % new_shape_prod != 0:
            #     raise ValueError(
            #         "Incompatible products while inferring unknown dim.")
            # new_shape[unk_idx] = working_set_prod // new_shape_prod

        # Ensure compatibility.
        # if np.prod(new_shape) != np.prod(working_set):
        #     raise ValueError("Incompatible shape specification.")

        # Append prefix and suffix, and return.
        return bottom_shape[:start_idx] + new_shape + bottom_shape[end_idx:]

    def _name_scope_builder(self, name):
        """Builds a re-enterable name scope.

        Args:
            name: Desired name scope. Type `str`.

        Returns:
            A name scope string.
        """
        cur_name = tf.get_variable_scope().name
        if cur_name:
            return "%s/%s/" % (cur_name.rstrip("/"), name.rstrip("/"))
        return "%s/" % name.rstrip("/")
