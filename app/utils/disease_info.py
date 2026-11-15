"""Structured disease information shown with a prediction (description, symptoms, causes,
prevention, management). Classes without an entry here fall back to utils.disease.disease_dic.
The prediction itself always comes from the model; this module only holds reference text."""

DISEASE_INFO = {
    'Tomato___Bacterial_spot': {
        'description': 'A bacterial disease of tomato and pepper that spots leaves, stems and fruit and can cause heavy leaf loss in warm, wet weather.',
        'symptoms': [
            'Small, dark, water-soaked spots on leaves, sometimes with a yellow halo; spots may merge and leaves turn yellow and drop.',
            'Raised, scabby brown spots on green fruit.',
        ],
        'causes': [
            'Bacteria of the genus Xanthomonas, carried on infected seed and transplants and in crop debris.',
            'Spread by splashing rain, overhead irrigation and handling wet plants; favoured by warm, humid weather.',
        ],
        'prevention': [
            'Use certified disease-free seed and healthy transplants.',
            'Avoid overhead irrigation and working among plants while they are wet.',
            'Rotate away from tomato and pepper for at least two to three years and remove crop debris after harvest.',
        ],
        'management': [
            'Remove badly affected leaves or plants.',
            'Copper-based bactericides can slow spread when applied early as a protectant; follow local recommendations and label directions.',
        ],
    },
    'Tomato___Early_blight': {
        'description': 'A common fungal disease that starts on the older, lower leaves and works its way up the plant, reducing yield when many leaves are lost.',
        'symptoms': [
            'Brown spots with concentric rings ("target" pattern), often with a yellow halo, first on lower leaves.',
            'Dark, sunken lesions on stems and dark leathery patches at the stem end of fruit.',
        ],
        'causes': [
            'The fungus Alternaria (mainly A. solani / A. linariae), which survives in soil and infected plant debris.',
            'Spread by wind and splashing water; favoured by warm, humid weather and stressed or poorly fed plants.',
        ],
        'prevention': [
            'Rotate crops and remove or bury tomato debris after harvest.',
            'Mulch the soil and stake plants to keep leaves off the ground and reduce splash.',
            'Water at the base of plants and keep plants well nourished.',
        ],
        'management': [
            'Remove infected lower leaves as soon as spots appear.',
            'Apply a protectant fungicide registered for early blight (for example chlorothalonil, mancozeb or copper) according to the label.',
        ],
    },
    'Tomato___Late_blight': {
        'description': 'A fast-spreading and destructive disease of tomato and potato that can kill plants within days in cool, wet weather.',
        'symptoms': [
            'Large, irregular, greasy grey-green to dark brown patches on leaves, often starting at leaf edges.',
            'White, fuzzy growth on the underside of leaves in humid conditions; dark lesions on stems.',
            'Firm, greasy brown blotches on fruit.',
        ],
        'causes': [
            'The water mould Phytophthora infestans, whose airborne spores can travel long distances from infected tomato or potato crops.',
            'Favoured by cool, wet, humid weather with long periods of leaf wetness.',
        ],
        'prevention': [
            'Plant resistant varieties and healthy transplants.',
            'Destroy volunteer potato plants and cull piles, which can carry the disease.',
            'Space plants for good air movement and avoid overhead irrigation.',
        ],
        'management': [
            'Remove and destroy infected plants promptly; do not compost them.',
            'When late blight is reported nearby, apply a protectant fungicide registered for late blight according to the label.',
        ],
    },
    'Tomato___Leaf_Mold': {
        'description': 'A fungal disease mainly of greenhouse and tunnel tomatoes grown in high humidity.',
        'symptoms': [
            'Pale green to yellow patches on the upper side of leaves.',
            'Olive-green to greyish-brown velvety mould on the underside of the same patches; leaves wither and drop.',
        ],
        'causes': [
            'The fungus Passalora fulva (also called Fulvia fulva).',
            'Favoured by relative humidity above about 85% and poor ventilation.',
        ],
        'prevention': [
            'Ventilate greenhouses and tunnels and avoid overcrowding plants.',
            'Water early in the day at the base of plants so leaves stay dry.',
            'Grow resistant varieties where leaf mold is a recurring problem.',
        ],
        'management': [
            'Remove infected leaves and reduce humidity.',
            'Apply a fungicide registered for leaf mold if the disease continues to spread.',
        ],
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'A fungal leaf-spot disease that can strip the lower canopy, exposing fruit to sunscald and reducing yield.',
        'symptoms': [
            'Many small, round spots with dark brown edges and grey or tan centres, starting on lower leaves.',
            'Tiny black dots (fungal fruiting bodies) inside the spots; affected leaves yellow and fall.',
        ],
        'causes': [
            'The fungus Septoria lycopersici, which survives on tomato debris and related weeds.',
            'Spread by splashing water, tools and hands; favoured by warm, wet weather.',
        ],
        'prevention': [
            'Rotate crops, remove tomato debris and control nightshade weeds.',
            'Mulch, stake plants and avoid wetting the foliage.',
        ],
        'management': [
            'Remove spotted lower leaves early.',
            'Apply a protectant fungicide registered for Septoria leaf spot (for example chlorothalonil, mancozeb or copper) according to the label.',
        ],
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Damage by the two-spotted spider mite, a tiny pest that feeds on the underside of leaves and multiplies quickly in hot, dry conditions.',
        'symptoms': [
            'Fine yellow or white speckling (stippling) on leaves that later turn bronze and dry out.',
            'Fine webbing and tiny moving mites on the underside of leaves.',
        ],
        'causes': [
            'The mite Tetranychus urticae.',
            'Favoured by hot, dry, dusty conditions and by insecticides that kill the mites\' natural enemies.',
        ],
        'prevention': [
            'Avoid drought stress and keep dust down.',
            'Check the underside of leaves regularly, especially in hot, dry spells.',
            'Avoid unnecessary broad-spectrum insecticides so predatory mites and insects survive.',
        ],
        'management': [
            'Spray the underside of leaves with a strong jet of water to knock mites off.',
            'Use insecticidal soap or horticultural oil, or a registered miticide for heavy infestations, following the label.',
        ],
    },
    'Tomato___Target_Spot': {
        'description': 'A fungal disease of warm, humid regions that causes leaf spots and leaf drop and can also mark the fruit.',
        'symptoms': [
            'Brown leaf spots that enlarge with faint concentric rings and lighter centres, similar to early blight.',
            'Small, sunken brown spots on fruit; heavy infection causes leaf drop.',
        ],
        'causes': [
            'The fungus Corynespora cassiicola, which survives on crop debris and many other host plants.',
            'Favoured by warm temperatures, high humidity and long periods of leaf wetness.',
        ],
        'prevention': [
            'Rotate crops and remove crop debris.',
            'Prune and space plants for good air movement and avoid overhead irrigation.',
        ],
        'management': [
            'Remove heavily spotted leaves.',
            'Apply a fungicide registered for target spot on tomato according to the label.',
        ],
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'A viral disease spread by whiteflies that stunts plants and can cause severe yield loss, especially when young plants are infected.',
        'symptoms': [
            'Leaves curl upwards, stay small and turn yellow at the edges.',
            'Stunted, bushy plants with many flowers dropping before fruit set.',
        ],
        'causes': [
            'Tomato yellow leaf curl virus (a begomovirus).',
            'Transmitted by the whitefly Bemisia tabaci; it does not spread through seed or by handling.',
        ],
        'prevention': [
            'Plant resistant or tolerant varieties.',
            'Raise seedlings under insect-proof netting and control whiteflies early.',
            'Remove weeds and old crops that host whiteflies and the virus.',
        ],
        'management': [
            'There is no cure; remove and destroy infected plants to reduce spread.',
            'Control whitefly populations with integrated measures such as yellow sticky traps, reflective mulch and registered insecticides.',
        ],
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'A very stable virus that spreads easily by contact and reduces plant vigour, yield and fruit quality.',
        'symptoms': [
            'Light- and dark-green mottling (mosaic) on leaves, sometimes with curling or narrow, fern-like leaves.',
            'Stunted growth and uneven ripening or internal browning of fruit.',
        ],
        'causes': [
            'Tomato mosaic virus (a tobamovirus).',
            'Spread mechanically on hands, tools and clothing and through infected seed; it can survive for a long time in debris.',
        ],
        'prevention': [
            'Use certified seed and resistant varieties.',
            'Wash hands and disinfect tools and supports regularly when working with plants.',
        ],
        'management': [
            'There is no cure; remove and destroy infected plants.',
            'Disinfect tools, stakes and greenhouse surfaces after removing infected plants.',
        ],
    },
    'Potato___Early_blight': {
        'description': 'A common fungal leaf disease of potato that reduces the green leaf area available for tuber growth.',
        'symptoms': [
            'Dark brown spots with concentric rings ("target" pattern) on older, lower leaves, often limited by the leaf veins.',
            'Yellowing of the leaf tissue around the spots; badly affected leaves wither.',
        ],
        'causes': [
            'The fungus Alternaria solani, which survives in soil and potato debris.',
            'Favoured by warm weather with alternating wet and dry periods, and by plants stressed by poor nutrition or drought.',
        ],
        'prevention': [
            'Rotate potatoes with non-solanaceous crops and remove crop debris.',
            'Keep plants well fed (especially with nitrogen) and avoid drought stress.',
            'Grow less susceptible varieties where available.',
        ],
        'management': [
            'Apply a protectant fungicide registered for early blight on potato when spots first appear, following the label.',
            'Harvest tubers once the skins have set to reduce tuber infection.',
        ],
    },
    'Potato___Late_blight': {
        'description': 'The most destructive potato disease; it spreads rapidly in cool, wet weather and can also rot the tubers.',
        'symptoms': [
            'Pale green, water-soaked patches that quickly turn dark brown or black, often starting at leaf tips and edges.',
            'White, downy growth on the underside of leaves in humid conditions; blackened stems.',
            'Reddish-brown, dry rot just under the skin of infected tubers.',
        ],
        'causes': [
            'The water mould Phytophthora infestans, spread by airborne spores and from infected seed tubers and cull piles.',
            'Favoured by cool, wet weather with high humidity.',
        ],
        'prevention': [
            'Plant certified, disease-free seed tubers and resistant varieties.',
            'Destroy cull piles and volunteer potato plants.',
            'Hill up soil around plants to protect tubers from spores washing down.',
        ],
        'management': [
            'Remove and destroy infected plants promptly.',
            'Apply a protectant fungicide registered for late blight when weather favours the disease, following the label.',
            'Remove haulm (vines) two to three weeks before harvest to protect tubers.',
        ],
    },
    'Potato___healthy': {
        'description': 'No disease symptoms were recognised on this potato leaf.',
        'symptoms': [], 'causes': [],
        'prevention': [
            'Inspect plants regularly, especially in cool, wet weather when late blight spreads.',
            'Use certified seed tubers and rotate crops.',
        ],
        'management': [],
    },
    'Apple___Apple_scab': {
        'description': 'A fungal disease of apple leaves and fruit; severe infections cause early leaf drop and blemished, cracked fruit.',
        'symptoms': [
            'Olive-green to brown, velvety spots on leaves that later turn dark and may cause the leaf to curl or drop.',
            'Dark, scabby, corky spots on fruit, which may crack.',
        ],
        'causes': [
            'The fungus Venturia inaequalis, which overwinters in fallen infected leaves.',
            'Spores are released in spring rain; infection needs several hours of leaf wetness in cool to mild weather.',
        ],
        'prevention': [
            'Rake up and destroy or shred fallen leaves in autumn to reduce the source of spring infection.',
            'Prune trees to open the canopy so leaves dry quickly.',
            'Plant scab-resistant apple varieties.',
        ],
        'management': [
            'Apply a fungicide registered for apple scab from green tip through the spring infection period, following the label.',
        ],
    },
    'Apple___Cedar_apple_rust': {
        'description': 'A rust disease that needs both apple and juniper (eastern red cedar) to complete its life cycle.',
        'symptoms': [
            'Bright yellow to orange spots on the upper surface of apple leaves, often with a red border.',
            'Small cup-like structures with tube-shaped projections on the underside of the spots later in the season.',
        ],
        'causes': [
            'The fungus Gymnosporangium juniperi-virginianae.',
            'Spores from orange, jelly-like galls on nearby junipers infect apple leaves in wet spring weather.',
        ],
        'prevention': [
            'Plant rust-resistant apple varieties.',
            'Where practical, remove nearby junipers or the galls on them.',
        ],
        'management': [
            'Apply a fungicide registered for cedar apple rust from pink bud until shortly after petal fall when galls are active, following the label.',
        ],
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'A widespread fungal disease of maize leaves; it is usually minor but can reduce yield in susceptible hybrids.',
        'symptoms': [
            'Small, elongated, cinnamon-brown powdery pustules scattered on both the upper and lower leaf surfaces.',
            'Pustules turn dark brown to black late in the season; heavily infected leaves yellow and dry out.',
        ],
        'causes': [
            'The fungus Puccinia sorghi, whose spores are carried long distances by wind.',
            'Favoured by cool to moderate temperatures and high humidity or heavy dew.',
        ],
        'prevention': [
            'Plant resistant or tolerant hybrids.',
            'Plant early so the crop is well developed before rust pressure builds.',
        ],
        'management': [
            'Scout fields regularly; fungicides registered for common rust are usually only justified on susceptible hybrids when infection is heavy early in the season.',
        ],
    },
    'Grape___Black_rot': {
        'description': 'A fungal disease of grapevines that spots the leaves and can destroy the berries in warm, wet seasons.',
        'symptoms': [
            'Small, round, reddish-brown to tan spots on leaves with a dark border and tiny black dots inside.',
            'Berries turn brown, then shrivel into hard, black "mummies".',
        ],
        'causes': [
            'The fungus Guignardia bidwellii (Phyllosticta ampelicida), which overwinters in mummified berries and infected canes.',
            'Spread by rain splash; favoured by warm, wet weather.',
        ],
        'prevention': [
            'Remove mummified berries from vines and the ground, and prune out infected canes during dormancy.',
            'Train and prune vines for good air movement and sunlight.',
        ],
        'management': [
            'Apply a fungicide registered for black rot from early shoot growth until the berries are about four weeks past bloom, following the label.',
        ],
    },
    'Tomato___healthy': {
        'description': 'No disease symptoms were recognised on this tomato leaf.',
        'symptoms': [],
        'causes': [],
        'prevention': [
            'Keep inspecting plants regularly, especially the underside of lower leaves.',
            'Water at the base of plants, keep good spacing and rotate crops to keep diseases away.',
        ],
        'management': [],
    },
}
