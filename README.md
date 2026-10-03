# Crack Detection on Concrete Surfaces (SDNET2018)

A binary classifier that labels 256×256 px concrete tiles as **cracked** or **non-cracked**, built by fine-tuning an ImageNet-pretrained EfficientNet-B0.

The focus of this project is evaluation you can trust: a leakage-safe split, per-surface results, a leave-one-surface-out generalization test, and an error analysis of what the model gets wrong.

> **Live demo:** _add the Streamlit app link here_

## Highlights

- On photos held out from training, the model reaches **PR-AUC 0.890**, **precision 0.883**, **recall 0.786** and **F1 0.831** (threshold chosen on the validation set).
- A recall-oriented threshold reaches **90.5% recall at 46.4% precision**. Relative to the F1-optimal threshold, that finds about 160 more cracked tiles at the cost of about 1,260 more false alarms (roughly 8 per extra crack found).
- Performance drops on surface types the model never saw in training: leave-one-surface-out PR-AUC was 0.75 (Decks), **0.56 (Pavements)** and 0.81 (Walls).
- Stronger augmentation did not close the cross-surface gap on Pavements.
- Most errors sit on tiles with weak visual evidence (see [Error analysis](#error-analysis)).

## Dataset

[SDNET2018](https://www.kaggle.com/datasets/whoami-as/structural-defects-network-sdnet-2018) (Structural Defects Network): 56,092 image tiles of 256×256 px, cut from 230 larger photographs of concrete bridge decks, walls and pavements, labelled cracked or non-cracked.

| Surface | Source photos | Cracked tiles | Non-cracked tiles | % cracked |
|---|---|---|---|---|
| Decks | 54 | 2,025 | 11,595 | 14.9% |
| Pavements | 104 | 2,608 | 21,726 | 10.7% |
| Walls | 72 | 3,851 | 14,287 | 21.2% |
| **Total** | **230** | **8,484** | **47,608** | **15.1%** |

Because of the class imbalance, a model that always predicts "non-cracked" would score about 85% accuracy. All results below therefore use PR-AUC, precision, recall and F1 on the cracked class, not accuracy.

## Leakage-safe split

Each filename starts with a prefix that identifies the source photograph (for example `7001-115.jpg` is tile 115 of photo `7001`). Neighbouring tiles from the same photo look almost identical, so a random tile-level split would leak near-duplicates into the test set and inflate the scores.

Instead, all tiles from one photo are kept in the same split, using `StratifiedGroupKFold` to keep the cracked ratio similar across splits (seed 42, split saved to `sdnet_splits.csv`).

| Split | Tiles | Photos | % cracked |
|---|---|---|---|
| Train | 37,299 | 153 | 15.3% |
| Validation | 9,559 | 39 | 15.0% |
| Test | 9,234 | 38 | 14.5% |

All three surfaces appear in every split, and a check in the code asserts that no photo appears in more than one split.

## Model and training

- **Model:** EfficientNet-B0 pretrained on ImageNet, with the classifier replaced by a single-logit head.
- **Loss:** binary cross-entropy with a positive-class weight set to the square root of the negative-to-positive ratio.
- **Optimiser:** AdamW, learning rate 3e-4, weight decay 1e-4, batch size 64, mixed precision.
- **Augmentation:** horizontal and vertical flips, mild colour jitter.
- **Training:** 4 epochs on a Colab T4. The checkpoint with the best validation PR-AUC was kept (epoch 3: validation PR-AUC 0.8605; the four epochs scored 0.8401, 0.8565, 0.8605 and 0.8507).
- **Threshold:** chosen on the validation set (maximising F1), then applied once to the test set.

## Results

### Test set (9,234 tiles from 38 held-out photos)

| Operating point | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Best F1 on validation | 0.666 | 0.883 | 0.786 | 0.831 |
| Recall ≥ 90% on validation | 0.127 | 0.464 | 0.905 | 0.613 |

Test PR-AUC (threshold-free): **0.8902**.

Confusion matrix at threshold 0.666 (rows = true label, columns = prediction):

| | Predicted non-cracked | Predicted cracked |
|---|---|---|
| **Non-cracked** | 7,751 | 140 |
| **Cracked** | 288 | 1,055 |

The right operating point depends on the cost of a missed crack versus the cost of a human re-checking a flagged tile. The recall-oriented threshold transferred well from validation to test (target 90%, test recall 90.5%).

### By surface (all-surface model, test set)

| Surface | Test tiles | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Decks | 1,764 | 0.798 | 0.796 | 0.721 | 0.757 |
| Pavements | 4,446 | 0.918 | 0.915 | 0.803 | 0.855 |
| Walls | 3,024 | 0.903 | 0.890 | 0.795 | 0.840 |

## Generalization to unseen surface types

For each surface, the model was trained on the other two surfaces and tested on the whole held-out surface (3 epochs per fold, threshold chosen on the validation set of the seen surfaces).

| Held-out surface | Trained on | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Decks | Pavements + Walls | 0.754 | 0.738 | 0.642 | 0.687 |
| Pavements | Decks + Walls | **0.557** | 0.629 | 0.436 | 0.515 |
| Walls | Decks + Pavements | 0.811 | 0.916 | 0.587 | 0.715 |

Takeaways:

- **Pavements generalize worst.** A model that never saw pavements finds fewer than half of the cracked pavement tiles.
- **Walls is largely a threshold problem.** Precision is high (0.916) but recall is low (0.587), and PR-AUC is still 0.81, so the ranking of tiles is mostly intact while the threshold chosen on other surfaces does not fit.
- **Strong augmentation did not fix Pavements.** Adding random resized crops, heavy colour jitter, grayscale (p=0.5) and blur moved held-out Pavements PR-AUC from 0.557 to 0.595 and F1 from 0.515 to 0.509. This is a single run, so the small PR-AUC change may be noise.

## Error analysis

### Grad-CAM

Grad-CAM was run on the final convolutional block for missed cracks and false alarms (`gradcam_errors.png`). One hypothesis was that false alarms come from crack fragments at tile borders. To test it, the location of the Grad-CAM peak was checked for 100 sampled tiles per group (the outer 32 px band covers 44% of a tile, so that is the chance level):

| Group | Peak in outer 32 px band | Tiles analysed |
|---|---|---|
| True positives | 37% | 100 |
| Missed cracks | 36% | 92 |
| False alarms | 38% | 100 |

The three groups are indistinguishable, so **the border-fragment hypothesis is not supported**. For 8 of the 100 sampled missed cracks, Grad-CAM showed no positive evidence anywhere in the tile.

### Manual review

24 missed cracks and 24 false alarms were sampled at random (`missed_cracks_sheet.png`, `false_alarms_sheet.png`) and tagged by one reviewer. The categories are subjective and the sample is small, so this indicates patterns without proving them.

**Missed cracks (label = cracked, model said no)**

| Tag | Count |
|---|---|
| Clear crack visible | 2 |
| Faint or short mark, unsure | 5 |
| No crack visible at tile level | 17 |

**False alarms (label = non-cracked, model said cracked)**

| Tag | Count |
|---|---|
| Crack-like lines visible (label may be wrong) | 3 |
| Joint, seam, shadow edge or chipped surface | 4 |
| Plain concrete with pits, dots, stains or faint marks | 17 |

What this suggests:

- **Most missed cracks have weak visual evidence.** 10 of the 24 have a predicted probability below 0.10, and in 17 no crack was visible at tile level, which fits hairline cracks or tile-level labels that are ambiguous. The two clear misses are rough-textured pavement tiles.
- **Most false alarms are not explained by label noise.** Only 7 of 24 contain something that could justify the alarm. The other 17 are plain concrete with pits, dark dots or stains, often scored with high confidence (0.85 to 1.00), so the model does over-fire on those features.
- **Walls appear over-represented among sampled errors** (13 of 24 false alarms, 12 of 24 misses, against about a third of test tiles). With 24 samples per group this is only a pointer.

## Limitations

- Results come from **one split and one training run (one seed)**, so small differences between numbers are within plausible noise. The cross-surface experiments used 3 epochs, one fewer than the main model.
- The test set contains only **38 source photos**, so the metrics have wide uncertainty.
- Labels are **tile-level** and some are ambiguous (hairline cracks, cracks at tile boundaries). The model cannot localise a crack within a tile, and no pixel-level evaluation was done.
- The manual error review was done by **one reviewer on 24 tiles per group**.
- All data comes from one dataset. The model has not been tested on images from other cameras, regions or surface types, and it should not be used for real structural assessment.

## Demo and API

The model is served in two ways that share the same inference code (`app/model.py`). An uploaded photo is cut into non-overlapping 256×256 px tiles, each tile is classified on CPU, and the flagged tiles are outlined in the returned image.

- **Hosted demo:** a Streamlit app (`streamlit_demo/streamlit_app.py`) deployed on Streamlit Community Cloud, which is free.
- **API and container:** a FastAPI service with a Dockerfile. The image is built and the API tests run on every push through GitHub Actions. It is not hosted publicly, because free container hosting with enough memory for PyTorch was not available.

- `GET /` browser demo, `POST /predict` JSON API, `GET /health`, interactive API docs at `/docs`.
- Two operating points from the evaluation above: `balanced` (threshold 0.666) and `high_recall` (threshold 0.127).

```bash
# run locally
pip install -r requirements.txt
uvicorn app.main:app --port 7860          # then open http://localhost:7860

# or with Docker
docker build -t crack-detector .
docker run -p 7860:7860 crack-detector

# call the API
curl -X POST "http://localhost:7860/predict?mode=high_recall" -F "file=@photo.jpg"
```

The weights are expected at `weights/crack_effnetb0.pt` (override with the `MODEL_PATH` environment variable). Tests (`python -m pytest`) and a Docker build run on every push through GitHub Actions.

This is a research prototype. The model was trained on SDNET2018 tiles, photos from other cameras, distances or surface types may behave differently, and it must not be used for real structural assessment.

## Reproducing

1. Download SDNET2018 from Kaggle and unzip it so that `Decks/`, `Pavements/` and `Walls/` (each with `Cracked/` and `Non-cracked/`) are available.
2. Build a table of image paths with columns `surface`, `label` and `group` (surface + filename prefix before the dash).
3. Split with `StratifiedGroupKFold` using the group column (6 folds for test, then 5 folds on the remainder for validation, seed 42), or load the saved `sdnet_splits.csv`.
4. Fine-tune EfficientNet-B0 as described above and select the checkpoint and threshold on the validation set only.
5. Run the leave-one-surface-out experiment by repeating training with one surface held out.

Environment: Python 3, PyTorch, torchvision, scikit-learn, pandas, Pillow, matplotlib. Training was done on a Google Colab T4 GPU; Grad-CAM and inference run on CPU.

## Files

| File | Contents |
|---|---|
| `sdnet_splits.csv` | Train / validation / test split of every tile |
| `effnet_test_preds.csv` | Test-set probabilities and predictions |
| `loso_results.csv` | Leave-one-surface-out results |
| `loso_pavements_strongaug.csv` | Strong-augmentation rerun on held-out Pavements |
| `gradcam_errors.png` | Grad-CAM examples of missed cracks and false alarms |
| `missed_cracks_sheet.png`, `false_alarms_sheet.png` | Contact sheets used for the manual review |
| `app/` | FastAPI service (`main.py`), tile-wise inference (`model.py`) and the browser demo page |
| `streamlit_demo/` | Streamlit demo app and its dependencies, used for the hosted demo |
| `weights/crack_effnetb0.pt` | Trained EfficientNet-B0 weights |
| `Dockerfile`, `requirements.txt` | CPU-only container for the service |
| `tests/`, `.github/workflows/ci.yml` | API tests and the CI workflow |

## Acknowledgements

Dataset: SDNET2018 (Dorafshan, Thomas and Maguire, Utah State University). Model weights from torchvision.
