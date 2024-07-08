import mindspore.nn as nn
import mindspore.ops as ops


class TemporalTransformNetwork(nn.Cell):
    def __init__(self, prop_boundary_ratio, action_sample_num, start_sample_num, end_sample_num,
                 temporal_interval, norm_mode):
        super(TemporalTransformNetwork, self).__init__()
        self.temporal_interval = temporal_interval
        self.prop_boundary_ratio = prop_boundary_ratio
        self.action_sample_num = action_sample_num
        self.start_sample_num = start_sample_num
        self.end_sample_num = end_sample_num
        self.norm_mode = norm_mode

    def construct(self, segments, features, video_sec):
        s_len = segments[:, 1] - segments[:, 0]
        starts_segments = [segments[:, 0] - self.prop_boundary_ratio * s_len,
                           segments[:, 0]]
        starts_segments = ops.stack(starts_segments, axis=1)

        ends_segments = [segments[:, 1],
                         segments[:, 1] + self.prop_boundary_ratio * s_len]
        ends_segments = ops.stack(ends_segments, axis=1)

        starts_feature = self._sample_one_temporal(starts_segments, self.start_sample_num, features, video_sec)
        ends_feature = self._sample_one_temporal(ends_segments, self.end_sample_num, features, video_sec)
        actions_feature = self._sample_one_temporal(segments, self.action_sample_num, features, video_sec)
        return starts_feature, actions_feature, ends_feature

    def _sample_one_temporal(self, segments, out_len, features, video_sec):
        if self.norm_mode == 'padding':
            total_temporal_len = features.shape[2] * self.temporal_interval
            segments = (ops.clamp(segments / total_temporal_len, max=1., min=0.) - 0.5) / 0.5
        elif self.norm_mode == 'resize':
            segments = (ops.clamp(segments / video_sec.unsqueeze(-1), max=1., min=0.) - 0.5) / 0.5
        theta = segments.new_zeros((features.shape[0], 2, 3))
        theta[:, 1, 1] = 1.0
        theta[:, 0, 0] = (segments[:, 1] - segments[:, 0]) / 2.0
        theta[:, 0, 2] = (segments[:, 1] + segments[:, 0]) / 2.0
        features = features.unsqueeze(2)
        grid = ops.affine_grid(theta, (features.shape[0], features.shape[1], 1, out_len))
        stn_feature = ops.grid_sample(features, grid).view(features.shape[0], features.shape[1], out_len)
        return stn_feature
