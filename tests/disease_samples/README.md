# Harvestify disease detection – review image collection

28 real photographs of plant leaves across 10 classes of Harvestify's disease model, for
uploading through the Disease Detection page and for automated regression tests
(`tests/test_disease_regression.py`). These are internal project assets; they are not
served by the website.

> **Current folder contents:** this folder now holds an 11-image subset (one image per class, two for
> grape black rot). Several files were renamed to `_01` when the subset was made; `image_index.csv` and
> `prediction_results.csv` were reconciled by SHA-256 content hash, and the `original_filename` column
> records each image's name in the full set. All 28 original images remain in
> `../Harvestify_Disease_Review_Images.zip`. The figures below refer to the full 28-image set.

## Files

| File | Contents |
|---|---|
| `*.jpg` | The images, named `<Crop>_<Disease>_<NN>.jpg` |
| `image_index.csv` | Filename, crop, disease, model class, origin, source URL, licence, author, size, verification notes |
| `prediction_results.csv` | What the current ResNet9 model actually predicted for every image (see below) |

## Classes (exact model class names and output indices)

| Images | Model class | Index |
|---|---|---|
| Tomato_Early_Blight_01–03 | `Tomato___Early_blight` | 29 |
| Tomato_Late_Blight_01–03 | `Tomato___Late_blight` | 30 |
| Tomato_Healthy_01–03 | `Tomato___healthy` | 37 |
| Potato_Early_Blight_01–03 | `Potato___Early_blight` | 20 |
| Potato_Late_Blight_01–03 | `Potato___Late_blight` | 21 |
| Potato_Healthy_01 | `Potato___healthy` | 22 |
| Apple_Scab_01–03 | `Apple___Apple_scab` | 0 |
| Apple_Cedar_Rust_01–03 | `Apple___Cedar_apple_rust` | 2 |
| Corn_Common_Rust_01–03 | `Corn_(maize)___Common_rust_` | 8 |
| Grape_Black_Rot_01–03 | `Grape___Black_rot` | 11 |

Only one healthy potato image is included: no further suitable, correctly labelled photograph
of a healthy potato leaf was found in the sources below.

## Sources and licences

Automated downloading from Google Images is blocked and is not permitted by Google's terms, so
images come from two legitimate public sources:

* **Wikimedia Commons** (16 images) – each file page (in `source_url`) states the author and licence
  (public domain, CC0, CC BY or CC BY-SA). Many come from university plant-pathology and extension collections.
* **PlantDoc dataset** (12 images, CC BY 4.0; Singh et al., 2020, github.com/pratikkayal/PlantDoc-Dataset),
  whose images were originally gathered by web image search. The original web page of each image is not recorded by the dataset.

If these images are shown in slides or reports, credit the author and licence listed in `image_index.csv`.

## How the images were checked

1. Candidates were found by pathogen and crop name, then filtered by their descriptions (host plant,
   no fruit, tubers, microscopy, diagrams or catalogue scans).
2. **Every image was inspected visually** for the symptoms of its disease (e.g. concentric-ring lesions for
   early blight, orange spots with yellow halos for cedar apple rust, elongated cinnamon pustules for
   common rust). Images with stock-photo watermarks (Alamy, Fotolia), copyright stamps, unclear symptoms or
   doubtful labels were rejected.
3. PlantDoc labels were not trusted on their own: several "potato early blight" images were rejected because
   they showed tomato leaves, a bell-pepper fruit or late-blight lesions, and one "apple rust" file was an apple-scab image.
4. One Commons file named "Potato-leaf.jpg" was rejected because its description shows it is a tomato variety with potato-like leaves.

The labels are the best visual judgement of a non-specialist plus the source descriptions; they are not
laboratory-confirmed diagnoses.

## Model results on these images

All 28 images were uploaded to the running Flask application (`POST /disease-predict/analyze`).
Image decoding and ResNet9 inference succeeded for **28/28**. Predictions are in `prediction_results.csv`:

* exact class correct: **6 / 28**
* correct crop: **10 / 28**
* tomato early/late blight: 5 / 6 correct; the other classes are mostly misclassified, often with high confidence.

This is consistent with the separate evaluation on 2,485 PlantDoc field photos (9.2 % accuracy) and reflects
the model, not the images: it was trained only on PlantVillage photos of single leaves on plain backgrounds.
These images therefore show how the application behaves on real photographs; they do **not** demonstrate
that the model generalises, and they must not be used as evidence of accuracy.
