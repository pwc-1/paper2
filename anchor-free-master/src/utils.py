# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import ipdb


def dump_metric_result(data_path_pair: tuple):
    df, path = data_path_pair
    df.to_csv(path, index=False)


def bbox_xw_transform_inv(boxes, deltas, dx_w, dw_w):
    widths = boxes[:, 1] - boxes[:, 0]
    ctr_x = boxes[:, 0] + 0.5 * widths

    dx = deltas[:, 0] * dx_w
    dw = deltas[:, 1] * dw_w

    pred_ctr_x = dx * widths + ctr_x
    pred_w = ops.exp(dw) * widths
    pred_boxes = deltas.copy()
    # x1
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    # x2
    pred_boxes[:, 1] = pred_ctr_x + 0.5 * pred_w

    return pred_boxes


def bbox_xw_transform_batch(ex_rois, gt_rois):
    ex_widths = ops.clamp(ex_rois[:, 1] - ex_rois[:, 0], min=0.00001)
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths

    gt_widths = ops.clamp(gt_rois[:, 1] - gt_rois[:, 0], min=0.00001)
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dw = ops.log(gt_widths / ex_widths)
    targets = ops.stack((targets_dx, targets_dw), axis=1)
    return targets


def bbox_se_transform_batch(ex_rois, gt_rois):
    ex_widths = ops.clamp(ex_rois[:, 1] - ex_rois[:, 0], min=0.00001)

    s_offset = gt_rois[:, 0] - ex_rois[:, 0]
    e_offset = gt_rois[:, 1] - ex_rois[:, 1]

    targets_s = s_offset / ex_widths
    targets_e = e_offset / ex_widths
    targets = ops.stack((targets_s, targets_e), axis=1)
    return targets


def bbox_se_transform_inv(boxes, deltas, dse_w):
    widths = boxes[:, 1] - boxes[:, 0]
    s_offset = deltas[:, 0] * widths * dse_w
    e_offset = deltas[:, 1] * widths * dse_w
    pred_boxes = deltas.copy()
    pred_boxes[:, 0] = boxes[:, 0] + s_offset
    pred_boxes[:, 1] = boxes[:, 1] + e_offset
    return pred_boxes


def batch_iou(proposals, gt_boxes):
    len_proposals = proposals[:, 1] - proposals[:, 0]
    int_xmin = ops.maximum(proposals[:, 0], gt_boxes[:, 0])
    int_xmax = ops.minimum(proposals[:, 1], gt_boxes[:, 1])
    inter_len = ops.clamp(int_xmax - int_xmin, min=0.)
    union_len = len_proposals - inter_len + gt_boxes[:, 1] - gt_boxes[:, 0]
    jaccard = inter_len / (union_len + 0.00001)
    return jaccard


def iou_matrix(proposals):
    p_num = proposals.size(0)
    p1 = proposals.unsqueeze(1).expand(-1, p_num, -1)
    p2 = proposals.unsqueeze(0).expand(p_num, -1, -1)

    int_xmin = ops.maximum(p1[:, :, 0], p2[:, :, 0])
    int_xmax = ops.minimum(p1[:, :, 1], p2[:, :, 1])
    inter_len = ops.clamp(int_xmax - int_xmin, min=0.)
    union_len = p1[:, :, 1] - p1[:, :, 0] - inter_len + p2[:, :, 1] - p2[:, :, 0]
    jaccard = inter_len / (union_len + 0.00001)
    return jaccard



def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    # ipdb.set_trace()
    jaccard = np.divide(inter_len, union_len)
    return jaccard