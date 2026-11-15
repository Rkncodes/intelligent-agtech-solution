# Importing essential libraries and modules

from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, send_from_directory
from flask_babel import Babel, gettext as _
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.disease_info import DISEASE_INFO
from utils.fertilizer import fertilizer_advice, CROP_GROUP
from utils.crop_explain import CropProfiles, explain as explain_crop
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from sklearn.ensemble import RandomForestClassifier
import os
import re
import json
import sqlite3
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

ORIGINAL_DISEASE_CLASSES = disease_classes          # class order of plant_disease_model.pth
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
ORIGINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.pth')
FOCUSED_MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_focused_9class.pth')
FOCUSED_META_PATH = os.path.join(MODEL_DIR, 'plant_disease_focused_9class.json')


def load_disease_model():
    """Use the focused 9-class model when it is installed, otherwise the original 38-class checkpoint.
    Returns (model, class names in output-index order, checkpoint path)."""
    if os.path.exists(FOCUSED_MODEL_PATH) and os.path.exists(FOCUSED_META_PATH):
        with open(FOCUSED_META_PATH, encoding='utf-8') as f:
            classes = json.load(f)['classes']
        unknown = [c for c in classes if c not in ORIGINAL_DISEASE_CLASSES]
        if unknown or len(set(classes)) != len(classes):
            raise RuntimeError(f"Invalid class list in {FOCUSED_META_PATH}: {unknown or 'duplicates'}")
        path = FOCUSED_MODEL_PATH
    else:
        classes, path = ORIGINAL_DISEASE_CLASSES, ORIGINAL_MODEL_PATH
    model = ResNet9(3, len(classes))
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))  # strict
    model.eval()
    return model, classes, path


disease_model, disease_classes, disease_model_path = load_disease_model()
print(f"Disease model: {os.path.basename(disease_model_path)} ({len(disease_classes)} classes)")


# Loading crop recommendation model (retrained from CSV to avoid old pickle incompatibility)

crop_recommendation_model = None
crop_feature_means = None


def init_crop_model():
    global crop_recommendation_model, crop_feature_means
    # Resolve the CSV relative to this file (project root/Data-processed), not the working directory
    crop_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'Data-processed', 'crop_recommendation.csv')
    try:
        crop_data = pd.read_csv(crop_csv_path)
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = crop_data[feature_cols]
        y = crop_data['label']
        crop_feature_means = X.mean()
        model = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        crop_recommendation_model = model
    except Exception as e:
        print(f"ERROR: crop recommendation model failed to initialize from {crop_csv_path}: {e!r}")
        crop_recommendation_model = None


def load_crop_profiles():
    """Per-crop conditions from the crop dataset, used only to explain recommendations."""
    try:
        return CropProfiles(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         'Data-processed', 'crop_recommendation.csv'))
    except Exception as e:
        print(f"ERROR: crop explanation data failed to load: {e!r}")
        return None


crop_profiles = load_crop_profiles()


# ------------------------- MARKETPLACE DB -----------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'marketplace.db')


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_marketplace_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            currency TEXT NOT NULL DEFAULT 'INR',
            tag TEXT,
            badge TEXT,
            highlight TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            customer_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            email TEXT,
            address_line1 TEXT NOT NULL,
            address_line2 TEXT,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            pincode TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            subtotal REAL NOT NULL,
            shipping REAL NOT NULL,
            total REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'PLACED',
            notes TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER,
            name TEXT NOT NULL,
            unit_price REAL NOT NULL,
            qty INTEGER NOT NULL,
            line_total REAL NOT NULL
        )
        """
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) as c FROM products")
    count = cur.fetchone()["c"]
    if count == 0:
        demo_products = [
            (
                "High‑Yield Wheat Seed Pack",
                "seeds",
                "Drought‑tolerant wheat variety optimized for nitrogen‑efficient soils.",
                1299,
                "INR",
                "N 80–100, P 40–60, K 40–60",
                "Seeds",
                "Best for rabi season",
            ),
            (
                "Balanced NPK 10‑10‑10 Blend",
                "fertilizer",
                "All‑purpose granular fertilizer for cereals, fruits and vegetables.",
                899,
                "INR",
                "Balanced NPK",
                "Fertilizer",
                "Good for most field crops",
            ),
            (
                "Wireless Soil Moisture Sensor",
                "tools",
                "Real‑time moisture and temperature monitoring with mobile alerts.",
                3499,
                "INR",
                "IoT, sensors",
                "Tool",
                "Ideal for drip irrigation",
            ),
            (
                "Soil Testing & Advisory",
                "services",
                "Professional soil test with a customized fertilizer and crop rotation plan.",
                1999,
                "INR",
                "Lab test + report",
                "Service",
                "Report in 3 working days",
            ),
            (
                "Hybrid Tomato Seed Mix",
                "seeds",
                "Disease‑resistant hybrids curated for high humidity regions.",
                749,
                "INR",
                "Tomato, hybrid",
                "Seeds",
                "High market demand",
            ),
            (
                "Organic Compost Bundle",
                "fertilizer",
                "Enriched organic compost to improve soil structure and microbial activity.",
                1099,
                "INR",
                "Organic",
                "Fertilizer",
                "Great for low‑P soils",
            ),
        ]
        cur.executemany(
            """
            INSERT INTO products (name, category, description, price, currency, tag, badge, highlight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            demo_products,
        )
        conn.commit()
    conn.close()


def _parse_cart_json(cart_json: str):
    try:
        data = request.get_json(silent=True) if cart_json is None else None
    except Exception:
        data = None

    if cart_json is None:
        cart_json = (data or {}).get("cart_json")

    if not cart_json:
        return []

    try:
        import json

        items = json.loads(cart_json)
        if not isinstance(items, list):
            return []
        normalized = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = it.get("id")
            name = (it.get("name") or "").strip()
            price = float(it.get("price") or 0)
            qty = int(it.get("qty") or 0)
            if not name or price <= 0 or qty <= 0:
                continue
            normalized.append(
                {
                    "id": pid,
                    "name": name,
                    "price": price,
                    "qty": qty,
                }
            )
        return normalized
    except Exception:
        return []


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: (prediction (string), confidence (float, 0-1 softmax probability))
    """
    image = Image.open(io.BytesIO(img)).convert('RGB')
    probs = disease_probabilities(image, model)
    # Pick index with highest probability
    conf, pred = torch.max(probs, dim=0)
    # Retrieve the class label
    return disease_classes[pred.item()], conf.item()


def disease_probabilities(image, model=disease_model):
    """Softmax probabilities over disease_classes for an RGB PIL image."""
    # Training used ToTensor() only on 256x256 images (no normalization),
    # so inference must not normalize either. CenterCrop keeps the input at
    # exactly 256x256: ResNet9's fixed pooling fails on aspect ratios >= 2:1.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0))
    return torch.softmax(logits, dim=1)[0]

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# Initialize models and marketplace DB
init_crop_model()
init_marketplace_db()

# ------------------------- I18N -----------------------------------------------

LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "mr": "मराठी",
}

app.config["BABEL_DEFAULT_LOCALE"] = "en"
def get_locale():
    # Priority: query param ?lang=hi -> cookie -> best match
    lang = request.args.get("lang")
    if lang in LANGUAGES:
        return lang
    cookie_lang = request.cookies.get("lang")
    if cookie_lang in LANGUAGES:
        return cookie_lang
    return request.accept_languages.best_match(list(LANGUAGES.keys())) or "en"


# Flask-Babel v4 uses constructor locale_selector
babel = Babel(app)

@babel.localeselector
def get_locale():
    return 'en'


@app.context_processor
def inject_lang():
    return {"LANGUAGES": LANGUAGES, "CURRENT_LANG": get_locale()}


@app.route("/set-language/<lang>")
def set_language(lang: str):
    if lang not in LANGUAGES:
        return redirect(request.referrer or url_for("home"))
    resp = redirect(request.referrer or url_for("home"))
    resp.set_cookie("lang", lang, max_age=60 * 60 * 24 * 365)
    return resp

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        if crop_recommendation_model is None:
            return render_template('try_again.html', title=title)
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        temperature = None
        humidity = None

        try:
            result = weather_fetch(city) if city else None
            if result is not None:
                temperature, humidity = result
        except Exception:
            temperature = None
            humidity = None
        weather = {'source': 'weather' if temperature is not None and humidity is not None else 'average', 'city': city}

        if temperature is None or humidity is None:
            if crop_feature_means is not None:
                temperature = float(crop_feature_means['temperature'])
                humidity = float(crop_feature_means['humidity'])
            else:
                temperature = 25.0
                humidity = 70.0

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        explanation = None
        if crop_profiles is not None:
            inputs = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
            explanation = explain_crop(final_prediction, inputs, crop_profiles, weather)
        return render_template('crop-result.html', prediction=final_prediction, title=title, explanation=explanation)

# render fertilizer recommendation result page


MAX_NUTRIENT_VALUE = 500   # input sanity bound; the crop dataset's largest value is 205


def load_nutrient_ranges():
    """Typical N, P, K range per crop, taken from Harvestify's crop dataset."""
    try:
        crop_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'Data-processed', 'crop_recommendation.csv')
        data = pd.read_csv(crop_csv)
        grouped = data.groupby('label')[['N', 'P', 'K']].agg(['min', 'max'])
        return {crop: {n: (float(grouped.loc[crop, (n, 'min')]), float(grouped.loc[crop, (n, 'max')])) for n in 'NPK'}
                for crop in grouped.index}
    except Exception as e:
        print(f"ERROR: nutrient ranges failed to load: {e!r}")
        return {}


nutrient_ranges = load_nutrient_ranges()


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'
    form = {k: (request.form.get(k) or '').strip() for k in ('nitrogen', 'phosphorous', 'pottasium', 'cropname')}

    errors = []
    values = {}
    for field, nutrient, label in (('nitrogen', 'N', _('Nitrogen (N)')), ('phosphorous', 'P', _('Phosphorus (P)')),
                                   ('pottasium', 'K', _('Potassium (K)'))):
        raw = form[field]
        if raw == '':
            errors.append(_('%(name)s is required.', name=label))
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(_('%(name)s must be a number.', name=label))
            continue
        if not np.isfinite(value) or value < 0:
            errors.append(_('%(name)s cannot be negative.', name=label))
        elif value > MAX_NUTRIENT_VALUE:
            errors.append(_('%(name)s looks implausibly high (maximum %(max)d).', name=label, max=MAX_NUTRIENT_VALUE))
        else:
            values[nutrient] = value

    crop = form['cropname'].lower()
    if not crop:
        errors.append(_('Please select a crop.'))
    elif crop not in nutrient_ranges or crop not in CROP_GROUP:
        errors.append(_('Please select a crop from the list.'))

    if not nutrient_ranges:
        errors.append(_('Fertilizer advice is temporarily unavailable. Please try again later.'))

    if errors:
        return render_template('fertilizer.html', title=title, errors=errors, form=form), 400

    _statuses, advice = fertilizer_advice(crop, values, nutrient_ranges[crop])
    return render_template('fertilizer-result.html', recommendation=advice, title=title)

# ------------------------- DISEASE ANALYSIS (upload workflow) -------------------------

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP'}
# Thresholds measured on 950 PlantVillage leaves, 2,485 PlantDoc field photos and 114 non-leaf images.
# They flag unsuitable or uncertain inputs; they do not guarantee that a prediction is correct.
BLANK_STD_THRESHOLD = 5          # lowest brightness std of any real leaf image: 7.8; flat images: 0
LOW_LIGHT_MEAN, OVEREXPOSED_MEAN = 25, 240   # real leaf images ranged 28-237 mean brightness
MIN_PLANT_FRACTION = 0.05        # flags 0.2% of leaf images and 54% of non-leaf images
LOW_CONFIDENCE = 0.70            # chosen on PlantDoc train: flags 0.9% of PlantVillage, ~40% of field photos
if disease_model_path == FOCUSED_MODEL_PATH:
    # the focused model ships its own threshold, chosen on its validation data
    with open(FOCUSED_META_PATH, encoding='utf-8') as _f:
        LOW_CONFIDENCE = json.load(_f).get('low_confidence_threshold', LOW_CONFIDENCE)


def validate_quality_config():
    """Fail at startup, not per request, if an image-quality threshold is missing or out of range.
    Brightness values are 0-255 (PIL pixels); fractions and confidence are 0-1."""
    def number(name, value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise RuntimeError(f"Invalid disease image-quality setting {name}={value!r}; expected a number in [{low}, {high}].")
    number('BLANK_STD_THRESHOLD', BLANK_STD_THRESHOLD, 0, 255)
    number('LOW_LIGHT_MEAN', LOW_LIGHT_MEAN, 0, 255)
    number('OVEREXPOSED_MEAN', OVEREXPOSED_MEAN, 0, 255)
    number('MIN_PLANT_FRACTION', MIN_PLANT_FRACTION, 0, 1)
    number('LOW_CONFIDENCE', LOW_CONFIDENCE, 0, 1)
    if LOW_LIGHT_MEAN >= OVEREXPOSED_MEAN:
        raise RuntimeError("LOW_LIGHT_MEAN must be lower than OVEREXPOSED_MEAN.")


validate_quality_config()


class UploadError(ValueError):
    pass


def split_label(label):
    """'Corn_(maize)___Common_rust_' -> ('Corn (maize)', 'Common rust')"""
    crop, disease = label.split('___')
    crop = crop.replace('_', ' ').replace(' ,', ',').strip()
    disease = disease.replace('_', ' ').strip()
    return crop, disease[0].upper() + disease[1:]


def load_upload(data):
    """Validate uploaded bytes and return an RGB PIL image, or raise UploadError."""
    if not data:
        raise UploadError(_('The selected file is empty.'))
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(_('The image is larger than 10 MB. Please choose a smaller image.'))
    try:
        probe = Image.open(io.BytesIO(data))
        image_format = probe.format
        probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise UploadError(_('This file is not a readable image, or it is corrupted.'))
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise UploadError(_('Unsupported image format. Please upload a JPG, PNG, WEBP or BMP image.'))
    return image.convert('RGB')


def image_quality(image):
    """Cheap checks for unusable inputs. Returns (blocking_error or None, [warnings])."""
    small = image.resize((128, 128))
    pixels = np.asarray(small, dtype=np.float32)
    gray = pixels.mean(axis=2)
    if gray.std() < BLANK_STD_THRESHOLD:
        return _('The image appears to be blank or a single flat colour, so it cannot be analysed.'), []

    warnings = []
    if min(image.size) < 128:
        warnings.append(_('The image resolution is very low; details of the leaf may be lost.'))
    if gray.mean() < LOW_LIGHT_MEAN:
        warnings.append(_('The image is very dark. Try a photo taken in better light.'))
    elif gray.mean() > OVEREXPOSED_MEAN:
        warnings.append(_('The image is very bright or overexposed.'))
    hsv = np.asarray(small.convert('HSV'), dtype=np.float32)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    plant_fraction = ((hue >= 20) & (hue <= 170) & (sat >= 0.15) & (val >= 0.15)).mean()
    if plant_fraction < MIN_PLANT_FRACTION:
        warnings.append(_('Little or no leaf-coloured area was found. The result is unlikely to be meaningful '
                          'unless the photo shows a plant leaf.'))
    return None, warnings


@app.route('/disease-predict/analyze', methods=['POST'])
def disease_analyze():
    """JSON endpoint used by the disease page to analyse images without reloading."""
    file = request.files.get('file')
    if file is None:
        return jsonify(ok=False, error=_('No image was uploaded.')), 400
    try:
        image = load_upload(file.read())
    except UploadError as e:
        return jsonify(ok=False, error=str(e)), 400

    blocking, warnings = image_quality(image)
    if blocking:
        return jsonify(ok=False, error=blocking), 422

    try:
        probs = disease_probabilities(image)
    except Exception as e:
        print("DISEASE ANALYSIS FAILED:", repr(e))
        return jsonify(ok=False, error=_('The model could not analyse this image. Please try another image.')), 500

    top = torch.topk(probs, 3)
    candidates = []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        crop, disease = split_label(disease_classes[i])
        candidates.append({'label': disease_classes[i], 'crop': crop, 'disease': disease, 'confidence': p})
    best = candidates[0]
    print("ANALYZE:", best['label'], f"({best['confidence']:.1%})", warnings)

    # disease_dic entries start with their own "Crop: ... Disease: ..." lines; the page shows those separately
    description = re.sub(r'^\s*<b>\s*Crop\s*</b>\s*:.*?Disease\s*:.*?<br/>', '',
                         disease_dic[best['label']], count=1, flags=re.S)
    return jsonify(
        ok=True,
        crop=best['crop'], disease=best['disease'], label=best['label'],
        healthy=best['label'].endswith('___healthy'),
        confidence=best['confidence'],
        low_confidence=best['confidence'] < LOW_CONFIDENCE,
        alternatives=candidates[1:],
        description_html=description,
        info=DISEASE_INFO.get(best['label']),
        warnings=warnings,
    )

def supported_crops():
    """Comma-separated crop names the active disease model can recognise."""
    crops = sorted({split_label(c)[0].replace('Pepper, bell', 'bell pepper').replace(' (including sour)', '')
                    .replace('Corn (maize)', 'corn').lower() for c in disease_classes})
    return ', '.join(crops[:-1]) + ' and ' + crops[-1] if len(crops) > 1 else crops[0]

# ------------------------- DISEASE REFERENCE GALLERY -------------------------
# Curated, labelled reference photos (tests/disease_samples). Labels come from image_index.csv,
# never from the model; this gallery is separate from live inference.

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'disease_samples')
REFERENCE_CLASSES = ['Apple___Apple_scab', 'Apple___Cedar_apple_rust', 'Grape___Black_rot',
                     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy']


def load_reference_gallery():
    """One indexed image per reference class (first listed in image_index.csv that exists on disk)."""
    import csv
    entries = {}
    try:
        with open(os.path.join(REFERENCE_DIR, 'image_index.csv'), encoding='utf-8') as f:
            for row in csv.DictReader(f):
                label = row['model_class']
                if label in REFERENCE_CLASSES and label not in entries \
                        and os.path.isfile(os.path.join(REFERENCE_DIR, row['filename'])):
                    info = DISEASE_INFO.get(label, {})
                    entries[label] = {
                        'label': label, 'filename': row['filename'], 'crop': row['crop'], 'disease': row['disease'],
                        'description': info.get('description', ''), 'symptoms': info.get('symptoms', []),
                        'prevention': info.get('prevention', []), 'management': info.get('management', []),
                        'source_url': row['source_url'], 'license': row['license'], 'author': row['author'],
                    }
    except Exception as e:
        print(f"ERROR: reference gallery failed to load: {e!r}")
    return [entries[c] for c in REFERENCE_CLASSES if c in entries]


@app.route('/disease-reference')
def disease_reference():
    return render_template('disease_reference.html', title='Harvestify - Disease Reference Gallery',
                           entries=load_reference_gallery())


@app.route('/disease-reference/image/<path:filename>')
def disease_reference_image(filename):
    # Serve only files listed in the gallery index, never arbitrary paths
    if filename not in {e['filename'] for e in load_reference_gallery()}:
        abort(404)
    return send_from_directory(REFERENCE_DIR, filename)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    print("ROUTE HIT")

    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        print("POST RECEIVED")

        if 'file' not in request.files:
            print("NO FILE FOUND")
            return redirect(request.url)

        file = request.files.get('file')

        if not file:
            print("EMPTY FILE")
            return render_template('disease.html', title=title, supported_crops=supported_crops())

        try:
            img = file.read()
            print("IMAGE READ SUCCESS")

            prediction, confidence = predict_image(img)
            print("PREDICTION:", prediction, f"({confidence:.1%})")

            prediction = Markup(str(disease_dic[prediction])) + Markup(
                '<br/><br/><b>{}</b>: {:.1f}%'
            ).format(_('Confidence'), confidence * 100)

            return render_template(
                'disease-result.html',
                prediction=prediction,
                title=title
            )

        except Exception as e:
            print("ERROR OCCURRED:", repr(e))
            return render_template('try_again.html', title=title), 400

    return render_template('disease.html', title=title, supported_crops=supported_crops())


@app.route('/marketplace')
def marketplace():
    title = 'Harvestify - Marketplace'
    conn = get_db_connection()
    products = conn.execute(
        "SELECT id, name, category, description, price, currency, tag, badge, highlight FROM products"
    ).fetchall()
    conn.close()
    return render_template('marketplace.html', title=title, products=products)


@app.route('/debug/crop-model')
def debug_crop_model():
    # Simple debug endpoint to confirm model availability
    return jsonify({"has_model": crop_recommendation_model is not None})


@app.route('/api/marketplace/products')
def marketplace_products_api():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, name, category, description, price, currency, tag, badge, highlight FROM products"
    ).fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    return jsonify(data)


@app.route('/marketplace/add', methods=['POST'])
def marketplace_add():
    name = (request.form.get('name') or '').strip()
    category = (request.form.get('category') or '').strip()
    description = (request.form.get('description') or '').strip()
    price_raw = (request.form.get('price') or '').strip()
    tag = (request.form.get('tag') or '').strip()

    if not name or not category or not price_raw:
        return redirect(url_for('marketplace'))

    try:
        price = float(price_raw)
    except ValueError:
        return redirect(url_for('marketplace'))

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO products (name, category, description, price, currency, tag, badge, highlight)
        VALUES (?, ?, ?, ?, 'INR', ?, 'Community', 'Farmer listing')
        """,
        (name, category, description, price, tag),
    )
    conn.commit()
    conn.close()
    return redirect(url_for('marketplace'))


@app.route('/marketplace/checkout', methods=['GET'])
def marketplace_checkout():
    title = 'Harvestify - Checkout'
    return render_template('checkout.html', title=title)


@app.route('/marketplace/checkout', methods=['POST'])
def marketplace_checkout_submit():
    cart_json = request.form.get('cart_json')
    cart_items = _parse_cart_json(cart_json)
    if not cart_items:
        return redirect(url_for('marketplace'))

    customer_name = (request.form.get('customer_name') or '').strip()
    phone = (request.form.get('phone') or '').strip()
    email = (request.form.get('email') or '').strip()
    address_line1 = (request.form.get('address_line1') or '').strip()
    address_line2 = (request.form.get('address_line2') or '').strip()
    city = (request.form.get('city') or '').strip()
    state = (request.form.get('state') or '').strip()
    pincode = (request.form.get('pincode') or '').strip()
    payment_method = (request.form.get('payment_method') or 'cod').strip()
    notes = (request.form.get('notes') or '').strip()

    if not customer_name or not phone or not address_line1 or not city or not state or not pincode:
        return render_template('checkout.html', title='Harvestify - Checkout', form_error="Please fill all required fields.")

    subtotal = sum(it["price"] * it["qty"] for it in cart_items)
    shipping = 0.0 if subtotal >= 1500 else 49.0
    total = subtotal + shipping

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO orders
        (customer_name, phone, email, address_line1, address_line2, city, state, pincode, payment_method, subtotal, shipping, total, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            customer_name,
            phone,
            email if email else None,
            address_line1,
            address_line2 if address_line2 else None,
            city,
            state,
            pincode,
            payment_method,
            subtotal,
            shipping,
            total,
            notes if notes else None,
        ),
    )
    order_id = cur.lastrowid

    for it in cart_items:
        line_total = it["price"] * it["qty"]
        cur.execute(
            """
            INSERT INTO order_items (order_id, product_id, name, unit_price, qty, line_total)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                it.get("id"),
                it["name"],
                it["price"],
                it["qty"],
                line_total,
            ),
        )

    conn.commit()
    conn.close()
    return redirect(url_for('marketplace_order', order_id=order_id))


@app.route('/marketplace/order/<int:order_id>')
def marketplace_order(order_id: int):
    conn = get_db_connection()
    order = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if order is None:
        conn.close()
        abort(404)
    items = conn.execute(
        "SELECT name, unit_price, qty, line_total FROM order_items WHERE order_id = ? ORDER BY id ASC",
        (order_id,),
    ).fetchall()
    conn.close()
    return render_template('order_confirmation.html', title='Harvestify - Order confirmed', order=order, items=items)


@app.route('/marketplace/orders')
def marketplace_orders():
    conn = get_db_connection()
    orders = conn.execute(
        "SELECT id, created_at, customer_name, total, status FROM orders ORDER BY id DESC LIMIT 25"
    ).fetchall()
    conn.close()
    return render_template('orders.html', title='Harvestify - My orders', orders=orders)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
