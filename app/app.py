# Importing essential libraries and modules

from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
from flask_babel import Babel, gettext as _
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
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

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model (retrained from CSV to avoid old pickle incompatibility)

crop_recommendation_model = None
crop_feature_means = None


def init_crop_model():
    global crop_recommendation_model, crop_feature_means
    try:
        crop_data = pd.read_csv('Data-processed/crop_recommendation.csv')
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = crop_data[feature_cols]
        y = crop_data['label']
        crop_feature_means = X.mean()
        model = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        crop_recommendation_model = model
    except Exception:
        crop_recommendation_model = None


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
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

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
babel = Babel(app, locale_selector=get_locale)


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

        return render_template('crop-result.html', prediction=final_prediction, title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    try:
        crop_name = str(request.form['cropname']).strip().lower()
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        base_dir = os.path.dirname(os.path.abspath(__file__))
        fert_path = os.path.join(base_dir, 'Data', 'fertilizer.csv')
        df = pd.read_csv(fert_path)
        crop_mask = df['Crop'].str.lower() == crop_name

        if not crop_mask.any():
            return render_template('try_again.html', title=title)

        nr = df.loc[crop_mask, 'N'].iloc[0]
        pr = df.loc[crop_mask, 'P'].iloc[0]
        kr = df.loc[crop_mask, 'K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            key = 'NHigh' if n < 0 else "Nlow"
        elif max_value == "P":
            key = 'PHigh' if p < 0 else "Plow"
        else:
            key = 'KHigh' if k < 0 else "Klow"

        response = Markup(str(fertilizer_dic[key]))
        return render_template('fertilizer-result.html', recommendation=response, title=title)
    except Exception:
        return render_template('try_again.html', title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


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
