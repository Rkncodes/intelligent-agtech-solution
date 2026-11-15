"""Fertilizer advice: per-nutrient status (low / adequate / high) and consistent recommendations.

Status is judged against the typical N, P and K range recorded for each crop in
Harvestify's crop dataset (Data-processed/crop_recommendation.csv). Values are on the
same relative scale as that dataset; they are not laboratory soil-test units.
"""
from markupsafe import Markup, escape

NUTRIENT_NAMES = {'N': 'Nitrogen (N)', 'P': 'Phosphorus (P)', 'K': 'Potassium (K)'}

NUTRIENT_ROLE = {
    'N': 'Nitrogen drives leaf and stem growth and gives leaves their green colour. '
         'Shortage shows as pale or yellowing older leaves and stunted growth.',
    'P': 'Phosphorus supports root development, flowering and seed or fruit set. '
         'Shortage shows as weak roots, dull or purplish leaves and delayed maturity.',
    'K': 'Potassium regulates water use, strengthens stems and improves disease resistance and '
         'produce quality. Shortage shows as scorched or yellow leaf edges and weak stems.',
}

CROP_GROUP = {
    'rice': 'cereal', 'maize': 'cereal', 'cotton': 'fibre', 'jute': 'fibre',
    'chickpea': 'legume', 'kidneybeans': 'legume', 'pigeonpeas': 'legume', 'mothbeans': 'legume',
    'mungbean': 'legume', 'blackgram': 'legume', 'lentil': 'legume',
    'pomegranate': 'fruit', 'banana': 'fruit', 'mango': 'fruit', 'grapes': 'fruit', 'watermelon': 'fruit',
    'muskmelon': 'fruit', 'apple': 'fruit', 'orange': 'fruit', 'papaya': 'fruit', 'coconut': 'plantation',
    'coffee': 'plantation',
}

CROP_DISPLAY = {'kidneybeans': 'kidney beans', 'pigeonpeas': 'pigeon peas', 'mothbeans': 'moth beans',
                'blackgram': 'black gram', 'mungbean': 'mung bean'}

# Why each nutrient matters for each crop group
CROP_NOTE = {
    ('cereal', 'N'): '{crop} is a heavy nitrogen feeder; nitrogen largely determines tillering, leaf area and grain yield.',
    ('cereal', 'P'): 'Phosphorus helps {crop} establish strong roots early and supports grain filling.',
    ('cereal', 'K'): 'Potassium helps {crop} resist lodging, drought stress and leaf diseases.',
    ('fibre', 'N'): '{crop} needs a steady nitrogen supply for vegetative growth and fibre yield.',
    ('fibre', 'P'): 'Phosphorus supports root growth and early establishment of {crop}.',
    ('fibre', 'K'): 'Potassium improves stem strength and fibre quality in {crop}.',
    ('legume', 'N'): '{crop} is a legume: root-nodule bacteria fix most of the nitrogen it needs, '
                     'so it requires far less fertilizer nitrogen than cereals.',
    ('legume', 'P'): 'Phosphorus is especially important for {crop}, because nodule formation and nitrogen fixation depend on it.',
    ('legume', 'K'): 'Potassium supports pod filling and drought tolerance in {crop}.',
    ('fruit', 'N'): 'Nitrogen supports canopy growth in {crop}, but too much favours leaves over flowers and fruit.',
    ('fruit', 'P'): 'Phosphorus supports flowering and fruit set in {crop}.',
    ('fruit', 'K'): 'Potassium has a strong influence on fruit size, sweetness, colour and shelf life in {crop}.',
    ('plantation', 'N'): 'Nitrogen sustains the new leaf growth that {crop} relies on for yield.',
    ('plantation', 'P'): 'Phosphorus supports root development and flowering in {crop}.',
    ('plantation', 'K'): 'Potassium is needed in large amounts by {crop} for yield and produce quality.',
}

HIGH_EFFECT = {
    'N': 'Excess nitrogen encourages soft, leafy growth at the expense of flowers, fruit or grain, '
         'makes plants more prone to lodging and pests, and the surplus can wash into groundwater.',
    'P': 'Excess phosphorus is not toxic to most crops, but it can reduce the uptake of zinc and iron, '
         'and phosphorus-rich soil that erodes pollutes nearby water.',
    'K': 'Excess potassium can reduce the uptake of magnesium and calcium, which may cause deficiencies of those nutrients.',
}


def _actions(nutrient, status, statuses, group):
    """Recommendations for one nutrient, chosen so they never work against the other nutrients' status."""
    other_high = {n for n, s in statuses.items() if s == 'high' and n != nutrient}
    acts = []
    if nutrient == 'N' and status == 'low':
        if group == 'legume':
            acts.append('Treat seed with the appropriate Rhizobium inoculant so the crop can fix its own nitrogen.')
            acts.append('If needed, apply only a small starter dose of nitrogen at sowing; heavy nitrogen '
                        'fertilizer suppresses nodulation in legumes.')
        else:
            acts.append('Apply a nitrogen fertilizer such as urea or ammonium sulphate, split into two or three '
                        'smaller doses during active growth rather than one large dose.')
            if 'P' not in other_high:
                acts.append('Work well-rotted farmyard manure or compost into the soil to build a longer-term nitrogen supply.')
            acts.append('Include a legume or green-manure crop in your rotation to add nitrogen naturally.')
    elif nutrient == 'N' and status == 'high':
        acts.append('Withhold nitrogen fertilizer and fresh manure for this crop until levels come down.')
        acts.append('If other nutrients are needed, use fertilizers that contain no nitrogen.')
        acts.append('After harvest, a cover crop can take up surplus nitrogen and reduce losses to groundwater.')
    elif nutrient == 'P' and status == 'low':
        if 'N' in other_high:
            acts.append('Use a phosphorus fertilizer without nitrogen, such as single superphosphate, rather than DAP.')
        else:
            acts.append('Apply a phosphorus fertilizer such as single superphosphate (SSP) or DAP.')
        acts.append('Place phosphorus close to the root zone at sowing or planting, where young roots can reach it.')
        acts.append('For organic management, rock phosphate or bone meal release phosphorus more slowly.')
        acts.append('Keep soil pH roughly between 6 and 7; phosphorus becomes poorly available in very acidic or alkaline soil.')
    elif nutrient == 'P' and status == 'high':
        acts.append('Stop applying phosphorus fertilizers and limit manure, which is rich in phosphorus.')
        acts.append('Use fertilizers that supply only the nutrients that are actually needed.')
        acts.append('Reduce soil erosion and runoff, for example with mulch or ground cover.')
    elif nutrient == 'K' and status == 'low':
        if group in ('fruit', 'plantation'):
            acts.append('Apply potash fertilizer; sulphate of potash (SOP) is often preferred for fruit and plantation crops.')
        else:
            acts.append('Apply a potash fertilizer such as muriate of potash (MOP) or sulphate of potash (SOP).')
        acts.append('Return crop residues and compost to the field; they recycle potassium back into the soil.')
    elif nutrient == 'K' and status == 'high':
        acts.append('Stop applying potash fertilizers until levels come down.')
        acts.append('Use fertilizers that contain no potassium for any other nutrients that are needed.')
        acts.append('Watch for signs of magnesium deficiency (yellowing between the veins of older leaves).')
    return acts


def nutrient_status(value, low, high):
    if value < low:
        return 'low'
    if value > high:
        return 'high'
    return 'adequate'


def fertilizer_advice(crop, values, ranges):
    """Build the advice for one crop.

    values: {'N': float, 'P': float, 'K': float}
    ranges: {'N': (min, max), 'P': (min, max), 'K': (min, max)} typical range for this crop
    Returns (statuses dict, Markup html).
    """
    crop_name = CROP_DISPLAY.get(crop, crop)
    crop_title = crop_name[0].upper() + crop_name[1:]
    group = CROP_GROUP[crop]
    statuses = {n: nutrient_status(values[n], *ranges[n]) for n in 'NPK'}
    badge = {'low': ('badge-warning', 'Low'), 'adequate': ('badge-success', 'Adequate'), 'high': ('badge-danger', 'High')}

    rows = ''.join(
        f'<tr><td>{NUTRIENT_NAMES[n]}</td><td>{values[n]:g}</td><td>{ranges[n][0]:g} – {ranges[n][1]:g}</td>'
        f'<td><span class="badge {badge[statuses[n]][0]}">{badge[statuses[n]][1]}</span></td></tr>'
        for n in 'NPK')
    html = [
        f'<h3 class="h5 mb-3">Fertilizer advice for {escape(crop_name)}</h3>',
        '<table class="table table-sm mb-4"><thead><tr><th>Nutrient</th><th>Your value</th>'
        f'<th>Typical range for {escape(crop_name)}</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>',
    ]

    problems = [n for n in 'NPK' if statuses[n] != 'adequate']
    if not problems:
        html.append('<p><b>All three nutrients are within the typical range for '
                    f'{escape(crop_name)}.</b> No corrective fertilizer is needed. Continue with balanced, '
                    'maintenance-level fertilization matched to the crop\'s growth stage, and recheck your soil '
                    'each season.</p>')
    for n in problems:
        status = statuses[n]
        direction = 'below' if status == 'low' else 'above'
        html.append(f'<h4 class="h6 mt-4">{NUTRIENT_NAMES[n]} is {status} '
                    f'({values[n]:g}, {direction} the typical {ranges[n][0]:g}–{ranges[n][1]:g})</h4>')
        html.append(f'<p class="mb-2">{NUTRIENT_ROLE[n]} {CROP_NOTE[(group, n)].format(crop=crop_title)}</p>')
        if status == 'high':
            html.append(f'<p class="mb-2">{HIGH_EFFECT[n]}</p>')
        html.append('<ul class="mb-0">' + ''.join(f'<li>{a}</li>' for a in _actions(n, status, statuses, group)) + '</ul>')
    adequate = [NUTRIENT_NAMES[n] for n in 'NPK' if statuses[n] == 'adequate']
    if problems and adequate:
        html.append(f'<p class="mt-4 mb-0">{" and ".join(adequate)} '
                    f'{"is" if len(adequate) == 1 else "are"} within the typical range; no change is needed.</p>')
    html.append('<p class="small text-muted mt-4 mb-0">Exact fertilizer quantities depend on your soil type, '
                'growth stage and local conditions. For application rates, follow a laboratory soil test report '
                'or the guidance of your local agricultural extension office.</p>')
    return statuses, Markup(''.join(html))
