import unittest

from sports.handball.config import CourtConfiguration, HandballCourtConfiguration


class HandballCourtConfigurationTest(unittest.TestCase):
    def test_default_court_dimensions(self):
        config = HandballCourtConfiguration()

        self.assertEqual(config.length, 4000)
        self.assertEqual(config.width, 2000)
        self.assertEqual(config.goal_width, 300)
        self.assertEqual(config.throw_off_area_radius, 200)
        self.assertEqual(config.free_throw_line_segment_length, 15)
        self.assertEqual(config.free_throw_line_gap_length, 15)

    def test_vertices_and_metadata_have_matching_lengths(self):
        config = CourtConfiguration()

        self.assertEqual(len(config.vertices), 39)
        self.assertEqual(len(config.labels), len(config.vertices))
        self.assertEqual(len(config.colors), len(config.vertices))

    def test_key_indexes_use_existing_vertices(self):
        config = CourtConfiguration()
        vertex_count = len(config.vertices)

        for indexes in [
            config.court_corner_indexes,
            config.left_goal_indexes,
            config.right_goal_indexes,
            config.left_goal_area_indexes,
            config.right_goal_area_indexes,
        ]:
            self.assertTrue(all(1 <= index <= vertex_count for index in indexes))

    def test_free_throw_line_intersects_sideline(self):
        config = CourtConfiguration()
        left_top_free_throw_vertex = config.vertices[18]
        right_top_free_throw_vertex = config.vertices[22]

        self.assertEqual(left_top_free_throw_vertex[1], 0)
        self.assertEqual(right_top_free_throw_vertex[1], 0)
        self.assertAlmostEqual(
            left_top_free_throw_vertex[0],
            config.length - right_top_free_throw_vertex[0],
        )


if __name__ == "__main__":
    unittest.main()
