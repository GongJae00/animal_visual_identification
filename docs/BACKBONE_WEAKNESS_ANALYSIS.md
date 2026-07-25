# Backbone Algorithmic Weakness Analysis

## 1. Visual Channel — DogFaceNet (FaceNet + ResNet)

### Training data
- 2,522 dog folders from web scrape (Zenodo), not fully curated; some contain
  multiple dogs, black-and-white, or non-face images.
- DogFaceNet_224resized: 1,393 dog classes after alignment + resizing.
- SachaDee ONNX variant claims 96.7% but no published protocol, test set, or
  confidence interval.

### Weaknesses

1. **Breed imbalance.** 2,522 identities from web scraping follows web
   popularity: labrador/poodle/german shepherd dominate; rare breeds (xolo,
   azawakh, sloughi) have 1–3 samples or none. The model's embedding space is
   shaped by popular-breed variance, not identity variance.

2. **Triplet loss saturation.** ResNet + triplet loss on 2,522 classes with
   ~5–15 images/class is a small-data regime. Semi-hard triplet mining will
   saturate quickly — the model learns to separate breeds, not individuals.
   Identity verification at 96.7% likely reflects breed-level separation, not
   true within-breed individual discrimination.

3. **Eye-nose alignment dependency.** DogFaceNet requires 3-point alignment
   (left eye, right eye, nose). Our video pipeline detects face via YOLO11
   bounding box, not 3-point landmarks. Misalignment → embedding degradation
   that was never tested.

4. **Web-scrape domain.** Training images are high-quality posed/"cute" photos.
   Deployment is video frames (motion blur, low light, partial face, side
   profile). Domain gap is severe and unmeasured.

5. **No open-set calibration.** FaceNet-style models minimize intra-class
   variance and maximize inter-class distance within the training set. There is
   no mechanism for detecting out-of-distribution dogs — every query matches
   something, including dogs never seen before. Threshold selection is
   entirely post-hoc on validation data.

6. **Fixed 224×224 input.** Small resolution discards fine-grained facial
   features (scar, whisker pattern, iris pigmentation) that humans use to
   distinguish individual dogs within a breed.

---

## 2. Texture Channel — ConvNeXt-Base finetuned on YT-BB-dog

### Model
- ConvNeXt-Base: 88M params, ImageNet-22K pretrained → ImageNet-1K finetuned.
- We plan to finetune on ~12K YT-BB-dog training oracle crops with ArcFace
  loss to produce a texture embedding.

### Weaknesses

1. **Texture = coat pattern, not identity.** A ConvNeXt trained on dog crops
   will learn coat color, fur length, and pattern (brindle, merle, spotted).
   Within a breed, coat variation is limited. Two black labradors have nearly
   identical texture embeddings. Texture alone is a weak identity signal for
   solid-color breeds (labrador, doberman, rhodesian ridgeback).

2. **Lighting invariance failure.** Coat color changes dramatically under
   different lighting (warm indoor → blue outdoor shadows). ConvNeXt is not
   inherently color-constancy aware. Two images of the same dog under different
   lights may be farther apart than two different dogs under the same light.

3. **Illumination normalization.** ImageNet preprocessing subtracts global
   mean/std; this does not correct for illuminant color. A brindle dog under
   sodium路灯 (yellow) will have different texture features than under
   daylight.

4. **Seasonal coat variation.** Dogs shed, change coat density, and some
   breeds change color seasonally (Siberian husky, Pyrenean). Texture
   embedding is inherently time-variant for the same individual.

5. **Crop alignment sensitivity.** The oracle crop is a face crop. Texture
   features extracted from a face crop capture fur direction, not coat
   pattern. Back fur (saddle, blanket) is the breed-identifying region but may
   be absent from face crops. We are using face-crop oracle crops (YT-BB-dog
   format), not body images.

6. **ArcFace on 12K samples.** 12K images across ~4K identities = 3
   images/identity on average. ArcFace with 4K classes × 768-d weight matrix
   (3M parameters in the classifier layer alone) will overfit severely.
   Regularization (label smoothing, weight decay) cannot fully compensate for
   the extremely low samples-per-class regime.

7. **Breed prior leaking.** ConvNeXt pretrained on ImageNet has strong breed
   priors (120 dog breeds in ImageNet-1K via Stanford Dogs overlap). Finetuning
   on YT-BB-dog may not override these priors — the embedding may still encode
   breed rather than identity.

---

## 3. Structural Channel — SuperAnimal-Quadruped HRNet-w32

### Model
- HRNet-w32, 17 keypoints, trained on Quadruped-80K (Stanford Dogs, AP-10K,
  AnimalPose, Horse-10, AcinoSet, iRodent).
- Zero-shot: no finetuning on our data.
- We derive a 256-d geometric embedding from pairwise distances and angles.

### Weaknesses

1. **17 keypoints are whole-body, not face-specific.** SuperAnimal-Quadruped
   outputs 17 body keypoints (nose, eyes, ears, shoulders, hips, tail, paws).
   Only ~6 keypoints are on the head. Dog identity information is concentrated
   in facial geometry (ear set, eye spacing, muzzle width, nose shape), which
   requires 30+ facial keypoints. DogFLW uses 46 facial landmarks. We are
   discarding 80% of facial structural information.

2. **Side-view bias.** SuperAnimal-Quadruped is trained primarily on
   orthogonal side-view images (quadruped body pose benchmark format). Our
   deployment uses frontal/3/4-face views from overhead cameras. The model was
   never evaluated on frontal dog face images. Keypoint accuracy will degrade
   substantially on out-of-view poses.

3. **Zero-shot failure on brachycephalic breeds.** Pugs, bulldogs, and
   boxers have foreshortened snouts → nose and eye keypoints cluster
   abnormally. The model was trained on a mix where labrador/golden
   retriever proportions dominate geometric priors. Extreme snout-length
   variation will push keypoints outside the learned distribution, producing
   implausible geometries.

4. **Geometric embedding degeneracy.** Pairwise distances+angles from 6 head
   keypoints produce n*(n-1) = 15 distances + 15 angles = 30 dims of facial
   information (out of 256 total; the rest is body geometry that has zero
   identity value in face-crop deployments). The effective identity signal is
   30 dims contaminated by ~226 dims of body-position noise.

5. **No self-occlusion handling.** The model outputs keypoints regardless of
   visibility. When a dog turns its head, ear and eye keypoints may be
   projected to implausible 2D locations. The geometric embedding will encode
   spurious distances that vary systematically with head pose, not identity.

6. **Scale ambiguity.** Keypoints are 2D pixel coordinates. Geometric
   features (distance between eyes) vary with head scale (dog distance from
   camera), not just identity. Our normalization (center + scale) helps but
   is fragile: self-occlusion changes the apparent center, and depth
   rotation changes apparent distances non-uniformly.

7. **No temporal consistency.** Video frames independently keypointed →
   geometric features jitter frame-to-frame. Identity should be temporally
   stable, but the structural embedding has no mechanism to enforce this.

---

## 4. Nose Print — Pet-ReID-IMAG (ResNeSt-101)

### Model
- ResNeSt-101, 2048-d embedding, trained on CVPR2022 Pet Biometric Challenge
  dataset.
- 91.7% AUC on phase A, 86.27% on phase B (harder set).

### Weaknesses

1. **Data scarcity.** CVPR2022 Pet Challenge dataset is small (~8K nose
   images, ~800 dogs). ResNeSt-101 (101 layers, 2048-d) will overfit to
   dataset-specific artifacts (background, lighting rig, cropping method).
   The competition test set uses a different capture setup than the training
   set, which is why phase B accuracy drops 5.4% — evidence of domain shift
   sensitivity.

2. **Nose capture requirement.** Requires a close-up, well-lit, centered
   snout image. In our deployment (overhead camera, 2–3m distance), the
   dog's nose occupies ~20–50 pixels — far too few for a 224×224 crop.
   Dedicated nose detection + super-resolution is needed, adding a complex
   preprocessing pipeline with its own failure modes (nostril detection,
   wet-nose reflection, shadow occlusion).

3. **Wet nose problem.** Dog noses are wet. Reflections and highlights
   create specular artifacts that shift the apparent texture. The competition
   dataset was likely captured with controlled lighting. In the wild, wet
   noses (post-drink, rain, panting) create non-biometric variance that
   overwhelms identity signal.

4. **Angle sensitivity.** Nose print texture varies with camera angle (frontal
   vs 45° vs 60°). The competition dataset is largely frontal close-ups. Our
   deployment cannot guarantee frontal nose presentation → matching a frontal
   gallery image to a 45° query requires pose normalization that does not
   exist.

5. **AUC ≠ accuracy at threshold.** 86–91% AUC measures ranking quality, not
   verification accuracy at a given threshold. For deployment, the relevant
   metric is TPR @ low FPR (e.g., 1% FAR). AUC of 86% in a balanced binary
   problem still means ~14% of pairs are incorrectly ordered. For identity
   verification with 1,000+ enrolled dogs, this error rate produces
   accumulate false positives that overwhelm the system.

6. **Long-term stability unknown.** Human nose prints change slowly with age.
   Canine nose print stability across the lifespan is unstudied. A puppy's
   nose texture may change as the dog matures, rendering enrolled templates
   stale within months.

7. **3D→2D projection loss.** Nose print is inherently a 3D texture (ridge
   pattern). 2D imaging loses depth information, making the representation
   sensitive to lighting direction (shadows emphasize different ridges).
   Photometric stereo or 3D capture would be more robust but is impractical
   in deployment.

---

## Cross-Cutting Weaknesses

### 8. Fusion naivety
Current fusion is weighted average of cosine similarities. This assumes:
- All channels are equally reliable across conditions (false: texture fails
  on solid-color dogs, structural fails on side views, nose fails on wet
  noses).
- Scores are calibrated on the same scale (false: each backbone outputs
  uncalibrated cosine values).
- Independence (false: visual and texture both depend on the same YOLO face
  crop).

### 9. No quality-aware gating
No channel can abstain when its input quality is poor. A blurry frame
produces a low-confidence embedding that is fused equally with a sharp one.

### 10. Missing temporal aggregation
Single-frame search is vulnerable to outlier frames (blink, yawn, motion).
No track-level median aggregation of embeddings.

### 11. Open-set blind spot
All channels produce nearest-neighbor matches, not open-set decisions.
There is no learned rejection threshold per channel or per identity.
