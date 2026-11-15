"""Regression tests for Harvestify disease detection.

Run from the project root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

The labelled fixtures are PlantVillage-style images (see fixtures/disease_samples/SOURCES.txt).
They are very probably part of the model's training data, so these tests guard against
pipeline regressions (loading, preprocessing, class mapping, API); they do NOT measure accuracy.
"""
import io
import json
import os
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, 'app')
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures', 'disease_samples')
sys.path.insert(0, APP_DIR)

import app as harvestify  # noqa: E402
from utils.disease import disease_dic  # noqa: E402


def png_bytes(array):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buf, 'PNG')
    return buf.getvalue()


class DiseaseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = harvestify.app.test_client()
        with open(os.path.join(FIXTURES, 'labels.json'), encoding='utf8') as f:
            cls.labels = json.load(f)

    def fixture_bytes(self, name):
        with open(os.path.join(FIXTURES, name), 'rb') as f:
            return f.read()

    def analyze(self, data, filename='leaf.png'):
        return self.client.post('/disease-predict/analyze', data={'file': (io.BytesIO(data), filename)},
                                content_type='multipart/form-data')

    # --- model and class mapping -------------------------------------------------------
    def test_model_matches_class_list(self):
        self.assertFalse(harvestify.disease_model.training)
        n = len(harvestify.disease_classes)
        self.assertEqual(harvestify.disease_model.classifier[2].out_features, n)
        self.assertEqual(len(set(harvestify.disease_classes)), n)
        self.assertEqual(len(harvestify.ORIGINAL_DISEASE_CLASSES), 38)
        self.assertTrue(set(harvestify.disease_classes) <= set(harvestify.ORIGINAL_DISEASE_CLASSES))
        if n != 38:   # focused model: class order must come from its metadata file
            with open(harvestify.FOCUSED_META_PATH, encoding='utf-8') as f:
                self.assertEqual(json.load(f)['classes'], harvestify.disease_classes)

    def test_every_class_has_a_description(self):
        # every class the active model can output has reference text
        self.assertTrue(set(harvestify.disease_classes) <= set(disease_dic))
        self.assertEqual(set(harvestify.ORIGINAL_DISEASE_CLASSES), set(disease_dic))
        if len(harvestify.disease_classes) == 9:
            self.assertTrue(set(harvestify.disease_classes) <= set(harvestify.DISEASE_INFO))

    def test_labelled_samples_still_predicted(self):
        for name, truth in self.labels.items():
            with self.subTest(sample=name):
                label, confidence = harvestify.predict_image(self.fixture_bytes(name))
                self.assertEqual(label, truth)
                self.assertGreater(confidence, 0.5)

    def test_wide_image_does_not_crash(self):
        leaf = Image.open(os.path.join(FIXTURES, 'TomatoHealthy1.JPG')).convert('RGB')
        wide = Image.new('RGB', (640, 256), leaf.getpixel((0, 0)))
        wide.paste(leaf, (192, 0))
        buf = io.BytesIO()
        wide.save(buf, 'PNG')
        label, _ = harvestify.predict_image(buf.getvalue())
        self.assertEqual(label, 'Tomato___healthy')

    # --- JSON analysis endpoint -------------------------------------------------------
    def test_analyze_returns_model_output(self):
        for name, truth in self.labels.items():
            with self.subTest(sample=name):
                data = self.fixture_bytes(name)
                r = self.analyze(data, name)
                self.assertEqual(r.status_code, 200)
                body = r.get_json()
                expected_label, expected_conf = harvestify.predict_image(data)
                self.assertTrue(body['ok'])
                self.assertEqual(body['label'], expected_label)
                self.assertAlmostEqual(body['confidence'], expected_conf, places=6)
                self.assertEqual(len(body['alternatives']), 2)
                self.assertTrue(body['description_html'])

    def test_analyze_rejects_bad_input(self):
        cases = {
            'empty': (b'', 400),
            'not an image': (b'this is not an image', 400),
            'truncated jpeg': (self.fixture_bytes('TomatoHealthy1.JPG')[:400], 400),
            'blank white': (png_bytes(np.full((256, 256, 3), 255)), 422),
            'blank black': (png_bytes(np.zeros((256, 256, 3))), 422),
        }
        for case, (data, status) in cases.items():
            with self.subTest(case=case):
                r = self.analyze(data)
                self.assertEqual(r.status_code, status)
                self.assertFalse(r.get_json()['ok'])
                self.assertTrue(r.get_json()['error'])

    def test_quality_config_is_validated(self):
        harvestify.validate_quality_config()          # current settings are valid
        for name, bad in (('BLANK_STD_THRESHOLD', None), ('BLANK_STD_THRESHOLD', 'five'), ('LOW_CONFIDENCE', 1.5),
                          ('MIN_PLANT_FRACTION', float('nan')), ('OVEREXPOSED_MEAN', 10)):
            with self.subTest(name=name, value=bad):
                original = getattr(harvestify, name)
                setattr(harvestify, name, bad)
                try:
                    with self.assertRaises(RuntimeError):
                        harvestify.validate_quality_config()
                finally:
                    setattr(harvestify, name, original)

    def test_small_images(self):
        # a 1x1 image is a single flat colour: rejected as blank
        self.assertEqual(self.analyze(png_bytes([[[30, 120, 40]]])).status_code, 422)
        # a tiny real leaf still reaches the model, with a resolution warning
        leaf = Image.open(os.path.join(FIXTURES, 'TomatoHealthy1.JPG')).convert('RGB').resize((48, 48))
        buf = io.BytesIO()
        leaf.save(buf, 'PNG')
        r = self.analyze(buf.getvalue())
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(body['ok'])
        self.assertTrue(any('resolution' in w for w in body['warnings']))

    def test_repeated_uploads_are_identical(self):
        data = self.fixture_bytes('TomatoEarlyBlight1.JPG')
        first, second = self.analyze(data).get_json(), self.analyze(data).get_json()
        self.assertEqual((first['label'], first['confidence']), (second['label'], second['confidence']))

    def test_review_images_run_through_the_model(self):
        # Real photographs in tests/disease_samples: checks decoding + inference only (accuracy is
        # reported in prediction_results.csv; field photos are often misclassified by this model).
        import csv
        review_dir = os.path.join(ROOT, 'tests', 'disease_samples')
        with open(os.path.join(review_dir, 'image_index.csv'), encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f) if os.path.exists(os.path.join(review_dir, r['filename']))]
        self.assertGreaterEqual(len(rows), 9)
        for row in rows:
            with self.subTest(image=row['filename']):
                self.assertIn(row['model_class'], harvestify.ORIGINAL_DISEASE_CLASSES)
                with open(os.path.join(review_dir, row['filename']), 'rb') as img:
                    r = self.analyze(img.read(), row['filename'])
                self.assertEqual(r.status_code, 200)
                body = r.get_json()
                self.assertIn(body['label'], harvestify.disease_classes)
                self.assertTrue(0.0 < body['confidence'] <= 1.0)

    def test_analyze_rejects_unsupported_format(self):
        buf = io.BytesIO()
        Image.new('RGB', (64, 64), (0, 128, 0)).save(buf, 'GIF')
        r = self.analyze(buf.getvalue(), 'leaf.gif')
        self.assertEqual(r.status_code, 400)

    def test_analyze_without_file(self):
        r = self.client.post('/disease-predict/analyze', data={})
        self.assertEqual(r.status_code, 400)

    # --- curated reference gallery (separate from live inference) ---------------------
    def test_reference_gallery(self):
        import csv
        import re
        page = self.client.get('/disease-reference')
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn('Curated Reference Example — Not a Live Model Prediction.', html)
        gallery = html[html.index('<section'):html.index('</section>')]
        self.assertNotRegex(gallery, r'\d+(\.\d+)?\s*%')                   # no confidence values
        self.assertIn('href="/disease-predict"', html)                     # navigation back
        entries = harvestify.load_reference_gallery()
        self.assertEqual([e['label'] for e in entries], harvestify.REFERENCE_CLASSES)
        review_dir = os.path.join(ROOT, 'tests', 'disease_samples')
        with open(os.path.join(review_dir, 'image_index.csv'), encoding='utf-8') as f:
            index = {r['filename']: r for r in csv.DictReader(f)}
        cards = re.findall(r'class="reference-crop">([^<]+)</span> – <span class="reference-disease">([^<]+)<', html)
        self.assertEqual(len(cards), 9)
        for e, (crop, disease) in zip(entries, cards):
            with self.subTest(entry=e['label']):
                row = index[e['filename']]
                self.assertEqual(row['model_class'], e['label'])               # image is the indexed one for this label
                self.assertEqual((crop, disease), (row['crop'], row['disease']))
                self.assertTrue(e['prevention'])                               # disease information present
                if not e['label'].endswith('healthy'):
                    self.assertTrue(e['symptoms'] and e['management'])
                r = self.client.get(f"/disease-reference/image/{e['filename']}")
                self.assertEqual(r.status_code, 200)
                with open(os.path.join(review_dir, e['filename']), 'rb') as img:
                    self.assertEqual(r.data, img.read())
                r.close()

    def test_reference_image_route_only_serves_gallery_files(self):
        for bad in ('Corn_Common_Rust_01.jpg', 'README.md', 'image_index.csv', '../test_fertilizer.py', 'nope.jpg'):
            with self.subTest(path=bad):
                self.assertEqual(self.client.get(f'/disease-reference/image/{bad}').status_code, 404)

    def test_detection_page_links_to_gallery(self):
        page = self.client.get('/disease-predict').get_data(as_text=True)
        self.assertIn('href="/disease-reference"', page)

    # --- page and classic form route -------------------------------------------------
    def test_page_has_upload_workflow_and_no_gallery(self):
        page = self.client.get('/disease-predict').get_data(as_text=True)
        self.assertIn('id="inputfile"', page)
        self.assertIn('Analyse Another Image', page)
        self.assertNotIn('Try a sample leaf', page)
        self.assertNotIn('/disease-predict/sample/', page)
        # static/css/style.css hides every element whose id starts with "drop" (mobile menu rule)
        self.assertNotRegex(page, r'id="drop')

    def test_classic_form_post_still_works(self):
        r = self.client.post('/disease-predict', data={'file': (io.BytesIO(self.fixture_bytes('PotatoEarlyBlight1.JPG')), 'x.jpg')},
                             content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        self.assertIn('Early Blight', r.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
