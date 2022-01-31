import numpy as np
import tensorflow as tf
import sys
from .tensor import pad_axis

def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.
    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        # print(idxes)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.
    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    # if isinstance(scores, dict) or isinstance(bboxes, dict):
    #     with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
    #         d_scores = {}
    #         d_bboxes = {}
    #         for c in scores.keys():
    #             s, b = bboxes_nms_batch(scores[c], bboxes[c],
    #                                     nms_threshold=nms_threshold,
    #                                     keep_top_k=keep_top_k)
    #             d_scores[c] = s
    #             d_bboxes[c] = b
    #         return d_scores, d_bboxes

    # Tensors inputs.
    s_shape = scores.get_shape().as_list()
    result_score = None
    result_box = None
    with tf.name_scope(scope, 'bboxes_nms_batch'):
      for i in range(s_shape[0]):
        r = bboxes_nms(scores[i,:], bboxes[i,:,:], nms_threshold, keep_top_k)
        tmp_scores, tmp_bboxes = r
        if result_score:
          result_score = tf.concat([result_score, tmp_scores])
          result_box = tf.concat([result_box, tmp_bboxes])
        else:
          result_score = tmp_scores
          result_box = tmp_bboxes
        # r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
        #                                    nms_threshold, keep_top_k),
        #               (scores, bboxes),
        #               dtype=(scores.dtype, bboxes.dtype),
        #               parallel_iterations=10,
        #               back_prop=False,
        #               swap_memory=False,
        #               infer_shape=True)
      # scores, bboxes = result[0]
      return result_score, result_box

def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.
    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    # if isinstance(scores, dict) or isinstance(bboxes, dict):
    #     with tf.name_scope(scope, 'bboxes_sort_dict'):
    #         d_scores = {}
    #         d_bboxes = {}
    #         for c in scores.keys():
    #             s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
    #             d_scores[c] = s
    #             d_bboxes[c] = b
    #         return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes, axis= 0)
            return [bb]
        r = fn_gather(bboxes, idxes)
        # r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
        #               [bboxes, idxes],
        #               dtype=[bboxes.dtype],
        #               parallel_iterations=10,
        #               back_prop=False,
        #               swap_memory=False,
        #               infer_shape=True)
        bboxes = r[0]
        return scores, bboxes
# import tfe
# =========================================================================== #
# SSD boxes selection.
# =========================================================================== #
def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=2, topk=300,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.
    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = predictions_layer.get_shape().as_list()
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], p_shape[-1]//2, -1]))
        l_shape = localizations_layer.get_shape().as_list()
        
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([1, l_shape[0],  l_shape[-1]]))
        d_scores = {}
        d_bboxes = {}

        for c in range(0, num_classes):
            if c == 0:
              continue
            # Remove boxes under the threshold.
            scores = predictions_layer[:,:, c]
            fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
            scores = scores * fmask
            bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
            # Append to dictionary.
            d_scores[c] = scores
            d_bboxes[c] = bboxes
        d_scores, d_bboxes = bboxes_nms_batch(d_scores.get(1), d_bboxes.get(1), nms_threshold=0.3, keep_top_k=300)
        d_scores, d_bboxes = bboxes_sort(d_scores, d_bboxes, top_k=100)
        # output = tf.concat([d_scores.get, d_bboxes], axis)
        return d_scores, d_bboxes

# =========================================================================== #
# TensorFlow implementation of boxes SSD decoding.
# =========================================================================== #
def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.
    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.
    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    prior_width = anchors_layer[:,2] - anchors_layer[:,0]
    prior_height = anchors_layer[:,3] - anchors_layer[:,1]
    prior_center_x = 0.5 * (anchors_layer[:,2] + anchors_layer[:,0])
    prior_center_y = 0.5 * (anchors_layer[:,3] + anchors_layer[:,1])
    decode_bbox_center_x = feat_localizations[:,0] * prior_width * prior_scaling[0]
    
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = feat_localizations[:, 1] * prior_width * prior_scaling[1]
    decode_bbox_center_y += prior_center_y

    decode_bbox_width = tf.exp(feat_localizations[:,2] * prior_scaling[2])
    decode_bbox_width *= prior_width
    decode_bbox_height = tf.exp(feat_localizations[:,3] * prior_scaling[3])
    decode_bbox_height *= prior_height
    
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
    bboxes = tf.stack([decode_bbox_xmin, decode_bbox_ymin,  decode_bbox_xmax, decode_bbox_ymax], axis=-1)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return bboxes


