import unittest

import numpy as np

from audio_engine import CARRIERS
from nearfield import ChirpProbe, NearField, PERIOD, RANGES
from spatial_imaging import ROOM_FRAMES, SpatialDemo, SpatialImager


class SpatialImagingTests(unittest.TestCase):
    def test_static_room_and_unconfirmed_paths_remain_blank(self):
        imager = SpatialImager(.15, 2)
        paths = np.ones((2, 2), dtype=bool)
        stereo = np.array([[0., .001], [.001, 0.]])
        room = np.zeros((2, 2, len(RANGES)), complex)
        room[:, :, 95] = .32
        for _ in range(ROOM_FRAMES+1):
            imager.update(room, paths, 1e-6, stereo)
        self.assertTrue(imager.ready)
        self.assertFalse(imager.image.any())

        point = np.array([.12, .29, .60])
        distance = (np.linalg.norm(imager.speakers-point, axis=1)[:, None]
                    + np.linalg.norm(imager.microphones-point, axis=1)[None]
                    - imager.direct_lengths + .15)/2
        echoes = room.copy()
        for i, j in np.ndindex(2, 2):
            nearest = abs(RANGES-distance[i, j]).argmin()
            echoes[i, j, nearest] += .23
        imager.update(echoes, paths, 1e-6, stereo)
        self.assertGreater(imager.image.max(), .3)
        px = abs(imager.x-point[0]).argmin()
        py = abs(imager.y-point[1]).argmin()
        self.assertGreater(imager.image[py-1:py+2, px-1:px+2].max(), .3)
        echoes[1, 0] = room[1, 0]
        imager.update(echoes, paths, 1e-6, stereo)
        self.assertGreater(imager.image[py-1:py+2, px-1:px+2].max(), .3)
        paths[0, :] = False
        imager.update(echoes, paths, 1e-6, stereo)
        self.assertFalse(imager.image.any())

    def test_pcm_drives_spatial_image_after_empty_room(self):
        imager = SpatialImager(.15, 2)
        probe = ChirpProbe(CARRIERS, multicarrier=True)
        near = NearField(2, CARRIERS, probe=probe, spatial=imager)
        demo = SpatialDemo(probe, imager)
        for i in range(62):
            near.analyze(demo.block(i*PERIOD, PERIOD), 110, (i+1)/25)
        self.assertTrue(imager.ready)
        self.assertFalse(imager.image.any())
        for i in range(62, 70):
            near.analyze(demo.block(i*PERIOD, PERIOD), 110, (i+1)/25)
        self.assertGreater(imager.image.max(), .25)
        self.assertGreater(np.count_nonzero(imager.image), 5)
        near.reset()
        self.assertFalse(imager.image.any())
        self.assertFalse(imager.ready)
