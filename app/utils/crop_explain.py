"""Plain-language explanation of a crop recommendation.

The Random Forest itself does not explain individual predictions. This module compares the
farmer's seven input values with the conditions recorded for the recommended crop in the same
dataset the model was trained on (Data-processed/crop_recommendation.csv). Nothing here is
written per crop: every range and comparison is computed from that data.
"""
import pandas as pd

FEATURES = [  # (column, label, unit)
    ('N', 'Nitrogen (N)', ''),
    ('P', 'Phosphorus (P)', ''),
    ('K', 'Potassium (K)', ''),
    ('temperature', 'Temperature', '°C'),
    ('humidity', 'Humidity', '%'),
    ('ph', 'Soil pH', ''),
    ('rainfall', 'Rainfall', 'mm'),
]
DISTINCT_Z = 0.75   # crop's typical value at least 0.75 standard deviations from the all-crop typical value

CROP_DISPLAY = {'kidneybeans': 'kidney beans', 'pigeonpeas': 'pigeon peas', 'mothbeans': 'moth beans',
                'blackgram': 'black gram', 'mungbean': 'mung bean', 'muskmelon': 'muskmelon'}


def crop_name(crop):
    return CROP_DISPLAY.get(crop, crop)


class CropProfiles:
    """Per-crop ranges and typical values computed from the training dataset."""

    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        cols = [f for f, _, _ in FEATURES]
        self.crop_min = data.groupby('label')[cols].min()
        self.crop_max = data.groupby('label')[cols].max()
        self.crop_median = data.groupby('label')[cols].median()
        self.all_min, self.all_max = data[cols].min(), data[cols].max()
        self.all_median, self.all_std = data[cols].median(), data[cols].std()
        self.n_crops = data['label'].nunique()


def _fmt(value, column):
    return f"{value:.1f}" if column in ('temperature', 'humidity', 'ph', 'rainfall') else f"{value:.0f}"


def explain(crop, inputs, profiles, weather):
    """Build the explanation for the predicted crop.

    inputs:  {'N':..., 'P':..., 'K':..., 'temperature':..., 'humidity':..., 'ph':..., 'rainfall':...}
    weather: {'source': 'weather' | 'average', 'city': str or None}
    """
    name = crop_name(crop)
    rows = []
    for col, label, unit in FEATURES:
        value = float(inputs[col])
        low, high = float(profiles.crop_min.loc[crop, col]), float(profiles.crop_max.loc[crop, col])
        status = 'below' if value < low else 'above' if value > high else 'within'
        z = (profiles.crop_median.loc[crop, col] - profiles.all_median[col]) / profiles.all_std[col]
        level = 'high' if z >= DISTINCT_Z else 'low' if z <= -DISTINCT_Z else None
        span = float(profiles.all_max[col] - profiles.all_min[col]) or 1.0
        pos = lambda v: max(0.0, min(100.0, (v - float(profiles.all_min[col])) / span * 100))
        rows.append({
            'column': col, 'label': label, 'unit': unit, 'value': _fmt(value, col),
            'low': _fmt(low, col), 'high': _fmt(high, col), 'status': status, 'level': level, 'z': abs(float(z)),
            'bar_left': pos(low), 'bar_width': max(1.5, pos(high) - pos(low)), 'marker': pos(value),
            'from_weather': col in ('temperature', 'humidity'),
        })

    within = [r for r in rows if r['status'] == 'within']
    outside = [r for r in rows if r['status'] != 'within']
    distinctive = sorted([r for r in rows if r['level']], key=lambda r: -r['z'])

    key_points = []
    for r in distinctive[:3]:
        phrase = r['label'].split(' (')[0]                       # 'Nitrogen (N)' -> 'Nitrogen'
        phrase = phrase.replace('Soil pH', 'soil pH') if 'pH' in phrase else phrase.lower()
        need = f"Fields growing {name} typically have {r['level']} {phrase} compared with most crops " \
               f"({r['low']}–{r['high']}{(' ' + r['unit']) if r['unit'] else ''})."
        if r['status'] == 'within':
            yours = f"Your value of {r['value']}{(' ' + r['unit']) if r['unit'] else ''} fits this."
        else:
            yours = f"Your value of {r['value']}{(' ' + r['unit']) if r['unit'] else ''} is {r['status']} this range."
        key_points.append({'label': r['label'], 'text': f"{need} {yours}", 'match': r['status'] == 'within'})

    if len(within) == len(rows):
        summary = (f"All seven of your values fall within the conditions recorded for {name}, "
                   f"so your field closely resembles the places where {name} is grown in Harvestify's data.")
    elif len(within) >= 5:
        summary = (f"{len(within)} of your 7 values fall within the conditions recorded for {name}. "
                   f"Overall, your field resembles where {name} is grown, but check the values marked outside the usual range.")
    else:
        summary = (f"Only {len(within)} of your 7 values fall within the conditions recorded for {name}. "
                   f"{name.capitalize()} was the closest match among the {profiles.n_crops} crops the model knows, "
                   f"but your conditions differ from its usual range in several ways, so treat this recommendation with caution.")

    advice = []
    for r in outside:
        unit = (' ' + r['unit']) if r['unit'] else ''
        advice.append(f"{r['label']} ({r['value']}{unit}) is {r['status']} the usual range for {name} "
                      f"({r['low']}–{r['high']}{unit}).")

    if weather['source'] == 'weather':
        weather_note = (f"Temperature and humidity were taken from the current weather for {weather['city']}. "
                        "The state and city are used only for this weather lookup.")
    else:
        weather_note = ("Current weather for your city could not be retrieved, so typical average values were used "
                        "for temperature and humidity. The state and city are used only for the weather lookup.")

    return {'crop': name, 'rows': rows, 'key_points': key_points, 'summary': summary,
            'outside': advice, 'weather_note': weather_note, 'within_count': len(within)}
