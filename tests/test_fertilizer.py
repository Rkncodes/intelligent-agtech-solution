"""Tests for the Fertilizer Advisor (status logic, consistency of advice, input validation).

Run from the project root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

import app as harvestify  # noqa: E402

# Phrases that must never appear for a given nutrient status
FORBIDDEN = {
    ('N', 'low'): ['Withhold nitrogen', 'sawdust', 'woodchip', 'sugar', 'leach'],
    ('N', 'high'): ['Apply a nitrogen fertilizer', 'manure or compost into the soil', 'Include a legume'],
    ('P', 'low'): ['Stop applying phosphorus'],
    ('P', 'high'): ['Apply a phosphorus fertilizer', 'rock phosphate', 'bone meal', 'farmyard manure'],
    ('K', 'low'): ['Stop applying potash'],
    ('K', 'high'): ['Apply potash', 'Apply a potash fertilizer'],
}
STATUS_TEXT = {'low': 'Low', 'adequate': 'Adequate', 'high': 'High'}


def statuses_from_html(html):
    rows = re.findall(r'<tr><td>(Nitrogen|Phosphorus|Potassium) \((\w)\)</td>.*?class="badge [\w-]+">(\w+)</span>', html)
    return {n: s.lower() for _, n, s in rows}


class FertilizerAdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = harvestify.app.test_client()

    def post(self, n, p, k, crop):
        return self.client.post('/fertilizer-predict', data={'nitrogen': n, 'phosphorous': p, 'pottasium': k, 'cropname': crop})

    def check(self, n, p, k, crop, expected):
        r = self.post(n, p, k, crop)
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertEqual(statuses_from_html(html), expected)
        for nutrient, status in expected.items():
            for phrase in FORBIDDEN.get((nutrient, status), []):
                self.assertNotIn(phrase, html, f'{nutrient} {status}: contradictory advice "{phrase}"')
            if status != 'adequate':
                self.assertIn(f'is {status} (', html)
        return html

    # rice typical ranges in the crop dataset: N 60-99, P 35-60, K 35-45
    def test_low_n_adequate_p_k(self):
        self.check(10, 50, 40, 'rice', {'N': 'low', 'P': 'adequate', 'K': 'adequate'})

    def test_user_reported_case(self):
        # N=10, P=50, K=50 for rice: N low, P adequate, K above rice's 35-45 range
        html = self.check(10, 50, 50, 'rice', {'N': 'low', 'P': 'adequate', 'K': 'high'})
        self.assertIn('urea', html)

    def test_low_p_only(self):
        self.check(80, 10, 40, 'rice', {'N': 'adequate', 'P': 'low', 'K': 'adequate'})

    def test_low_k_only(self):
        self.check(80, 40, 10, 'rice', {'N': 'adequate', 'P': 'adequate', 'K': 'low'})

    def test_all_low(self):
        self.check(5, 5, 5, 'banana', {'N': 'low', 'P': 'low', 'K': 'low'})

    def test_all_adequate(self):
        html = self.check(80, 40, 40, 'rice', {'N': 'adequate', 'P': 'adequate', 'K': 'adequate'})
        self.assertIn('No corrective fertilizer is needed', html)

    def test_excess_n(self):
        self.check(200, 40, 40, 'rice', {'N': 'high', 'P': 'adequate', 'K': 'adequate'})

    def test_excess_p(self):
        self.check(80, 150, 40, 'rice', {'N': 'adequate', 'P': 'high', 'K': 'adequate'})

    def test_excess_k(self):
        self.check(80, 40, 120, 'rice', {'N': 'adequate', 'P': 'adequate', 'K': 'high'})

    def test_multiple_excess(self):
        self.check(200, 150, 120, 'maize', {'N': 'high', 'P': 'high', 'K': 'high'})

    def test_low_n_with_high_p_avoids_manure(self):
        html = self.check(10, 150, 20, 'maize', {'N': 'low', 'P': 'high', 'K': 'adequate'})
        self.assertNotIn('farmyard manure', html)

    def test_high_n_with_low_p_avoids_dap(self):
        html = self.check(200, 10, 40, 'rice', {'N': 'high', 'P': 'low', 'K': 'adequate'})
        self.assertIn('rather than DAP', html)

    def test_crop_specific_advice(self):
        legume = self.check(0, 70, 20, 'lentil', {'N': 'adequate', 'P': 'adequate', 'K': 'adequate'})
        self.assertIn('lentil', legume)
        legume_low = self.check(0, 10, 20, 'chickpea', {'N': 'low', 'P': 'low', 'K': 'low'})
        self.assertIn('Rhizobium', legume_low)
        self.assertNotIn('urea', legume_low)
        fruit = self.check(20, 130, 50, 'grapes', {'N': 'adequate', 'P': 'adequate', 'K': 'low'})
        self.assertIn('fruit size', fruit)
        self.assertIn('sulphate of potash', fruit)
        cereal = self.check(10, 40, 20, 'maize', {'N': 'low', 'P': 'adequate', 'K': 'adequate'})
        self.assertIn('heavy nitrogen feeder', cereal)

    def test_every_crop_in_form_gives_advice(self):
        for crop in harvestify.nutrient_ranges:
            with self.subTest(crop=crop):
                r = self.post(50, 50, 50, crop)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(len(statuses_from_html(r.get_data(as_text=True))), 3)

    def test_invalid_inputs(self):
        cases = {
            'missing N': ('', 40, 40, 'rice', 'Nitrogen (N) is required'),
            'non-numeric P': (80, 'abc', 40, 'rice', 'Phosphorus (P) must be a number'),
            'negative K': (80, 40, -5, 'rice', 'Potassium (K) cannot be negative'),
            'implausible N': (5000, 40, 40, 'rice', 'implausibly high'),
            'no crop': (80, 40, 40, '', 'Please select a crop'),
            'unknown crop': (80, 40, 40, 'tomato', 'Please select a crop from the list'),
            'not a number': ('nan', 40, 40, 'rice', 'cannot be negative'),
        }
        for case, (n, p, k, crop, message) in cases.items():
            with self.subTest(case=case):
                r = self.post(n, p, k, crop)
                self.assertEqual(r.status_code, 400)
                html = r.get_data(as_text=True)
                self.assertIn(message, html)
                self.assertIn('name="nitrogen"', html)  # the form is shown again

    def test_missing_fields_entirely(self):
        r = self.client.post('/fertilizer-predict', data={})
        self.assertEqual(r.status_code, 400)

    def test_form_page_still_loads(self):
        r = self.client.get('/fertilizer')
        self.assertEqual(r.status_code, 200)
        self.assertIn('name="cropname"', r.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
