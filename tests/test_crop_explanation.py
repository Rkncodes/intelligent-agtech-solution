"""Tests for the crop recommendation explanation (prediction must be unchanged; explanation data-driven).

Run from the project root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

import app as harvestify  # noqa: E402
from utils.crop_explain import FEATURES  # noqa: E402

DATA = pd.read_csv(os.path.join(ROOT, 'Data-processed', 'crop_recommendation.csv'))


class CropExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = harvestify.app.test_client()

    def recommend(self, n, p, k, ph, rain, city=''):
        return self.client.post('/crop-predict', data={'nitrogen': n, 'phosphorous': p, 'pottasium': k,
                                                       'ph': ph, 'rainfall': rain, 'stt': '', 'city': city})

    def direct_prediction(self, n, p, k, ph, rain, temperature, humidity):
        return harvestify.crop_recommendation_model.predict(np.array([[n, p, k, temperature, humidity, ph, rain]]))[0]

    def test_prediction_unchanged_and_explanation_consistent(self):
        means = harvestify.crop_feature_means
        cases = [(90, 42, 43, 6.5, 202.9), (104, 18, 30, 6.7, 140.9), (20, 134, 200, 5.9, 112.7),
                 (40, 67, 80, 7.3, 80.0), (20, 20, 30, 5.8, 95.0)]
        crops_seen = set()
        for n, p, k, ph, rain in cases:
            with self.subTest(inputs=(n, p, k, ph, rain)):
                html = self.recommend(n, p, k, ph, rain).get_data(as_text=True)
                expected = self.direct_prediction(n, p, k, ph, rain, means['temperature'], means['humidity'])
                shown = re.search(r'You should grow <i>(\w+) </i>', html).group(1)
                self.assertEqual(shown, expected)                       # explanation does not alter the prediction
                crops_seen.add(shown)
                self.assertIn('Why is this crop recommended?', html)
                crop = DATA[DATA.label == shown]
                inputs = {'N': n, 'P': p, 'K': k, 'temperature': means['temperature'], 'humidity': means['humidity'],
                          'ph': ph, 'rainfall': rain}
                for col, label, _unit in FEATURES:
                    lo, hi = crop[col].min(), crop[col].max()
                    status = 'Below range' if inputs[col] < lo else 'Above range' if inputs[col] > hi else 'Within range'
                    row = re.search(rf'<td[^>]*>{re.escape(label)}(?:<sup>\*</sup>)?</td>.*?</tr>', html, re.S).group(0)
                    self.assertIn(status, row, f'{label} status')
                    fmt = (lambda v: f'{v:.1f}') if col in ('temperature', 'humidity', 'ph', 'rainfall') else (lambda v: f'{v:.0f}')
                    self.assertIn(f'{fmt(lo)} – {fmt(hi)}', row, f'{label} range')
        self.assertGreaterEqual(len(crops_seen), 4)                       # explanation is generic, not per-crop text

    def test_weather_values_are_used(self):
        original = harvestify.weather_fetch
        harvestify.crop_prediction.__globals__['weather_fetch'] = lambda city: (31.0, 85.0)
        try:
            html = self.recommend(90, 42, 43, 6.5, 202.9, city='Pune').get_data(as_text=True)
        finally:
            harvestify.crop_prediction.__globals__['weather_fetch'] = original
        self.assertIn('<b>31.0</b> °C', html)
        self.assertEqual(re.search(r'You should grow <i>(\w+) </i>', html).group(1),
                         self.direct_prediction(90, 42, 43, 6.5, 202.9, 31.0, 85.0))

    def test_removed_sections_are_absent(self):
        html = self.recommend(90, 42, 43, 6.5, 202.9).get_data(as_text=True)   # rainfall is outside jute's range
        self.assertNotIn('Worth checking', html)
        self.assertNotIn('About this explanation', html)
        self.assertIn('Above range', html)                      # statuses are still shown in the table
        self.assertIn('Key growing conditions for', html)

    def test_crop_form_still_loads(self):
        self.assertEqual(self.client.get('/crop-recommend').status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
