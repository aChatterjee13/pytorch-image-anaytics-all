import torch

from image_analytics.detection_3d import ops


class TestPointOps:
    def test_fps_shape_and_first_index(self):
        xyz = torch.rand(2, 128, 3)
        idx = ops.farthest_point_sample(xyz, 32)
        assert idx.shape == (2, 32)
        assert (idx[:, 0] == 0).all()  # deterministic start at index 0

    def test_fps_spreads_points(self):
        # Two tight clusters: FPS should pick from both, not just one.
        a = torch.zeros(50, 3)
        b = torch.ones(50, 3) * 10
        xyz = torch.cat([a, b]).unsqueeze(0)
        idx = ops.farthest_point_sample(xyz, 2)
        chosen = ops.index_points(xyz, idx)[0]
        # the two farthest points are one from each cluster
        assert (chosen[0] - chosen[1]).norm() > 15

    def test_knn_matches_bruteforce(self):
        torch.manual_seed(0)
        xyz = torch.rand(1, 40, 3)
        idx = ops.knn(5, xyz, xyz)
        # brute force nearest 5 for point 0
        d = ((xyz[0] - xyz[0, 0]) ** 2).sum(-1)
        expected = d.argsort()[:5].tolist()
        assert set(idx[0, 0].tolist()) == set(expected)

    def test_ball_query_within_radius(self):
        torch.manual_seed(0)
        xyz = torch.rand(1, 60, 3)
        new = xyz[:, :10]
        idx = ops.ball_query(0.3, 16, xyz, new)
        assert idx.shape == (1, 10, 16)
        # every grouped point within radius of its query (or a padded repeat)
        for q in range(10):
            grouped = xyz[0, idx[0, q]]
            assert ((grouped - new[0, q]) ** 2).sum(-1).sqrt().max() <= 0.3 + 1e-5

    def test_index_points_multi_dim(self):
        points = torch.arange(2 * 5 * 3).float().reshape(2, 5, 3)
        idx = torch.tensor([[0, 2], [1, 4]])
        out = ops.index_points(points, idx)
        assert out.shape == (2, 2, 3)
        assert torch.equal(out[0, 1], points[0, 2])
