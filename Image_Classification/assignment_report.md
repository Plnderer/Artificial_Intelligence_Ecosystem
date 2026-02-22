
# Image Classification and Processing Assignment Report (3.3)

**Student:** Eric Joel Reyes Rivera
**Repo Link:** [https://github.com/Plnderer/Artificial_Intelligence_Ecosystem/tree/main/Image_Classification](https://github.com/Plnderer/Artificial_Intelligence_Ecosystem/tree/main/Image_Classification)

---

## Part 1: Using the Basic Classifier and Implementing Grad CAM

### 3. Run the Basic Classifier

**Image Evaluated:** `Eric.png`

**Top-3 Predictions:**

1. `chain_saw` (0.12)
2. `flagpole` (0.08)
3. `assault_rifle` (0.07)

**Reflections on Code Explanation:**
The AI’s line by line explanation helped me understand the full classification pipeline. The most useful insight was that pretrained CNNs require consistent preprocessing: resizing the image to the model’s expected input dimensions (224×224), converting it into a numeric array, and expanding it into a batch dimension (`np.expand_dims`) before calling `predict()`. It also clarified how `decode_predictions()` translates raw output vectors into human-readable labels and confidence scores.

---

### 4. Implement Grad CAM

Grad CAM was integrated into `base_classifier.py` using TensorFlow’s `GradientTape`. The implementation:

* Targets the **last convolutional layer** (e.g., `out_relu` in MobileNetV2) to preserve spatial information.
* Computes gradients of the **top predicted class score** with respect to that layer’s feature maps.
* Weights and combines feature maps into a heatmap (with ReLU applied so only positive influence remains).
* Overlays the heatmap on the original image and saves an output image ending in `_cam.png`.

---

### 5. Understand Grad CAM

Grad CAM (Gradient-weighted Class Activation Mapping) is a visual explanation technique for CNN based models. It works by measuring how strongly different spatial regions in the final convolutional layer contribute to a chosen prediction. Areas shown in “hotter” colors represent regions that had the strongest positive impact on the model’s output for that class.

---

### 6. Analyze the Heatmap

For `Eric.png`, the model predicted `chain_saw` as the top class, but at a very low confidence (12%), followed by `flagpole` and `assault_rifle`, which suggests uncertainty and weak pattern matching.

**What the heatmap focused on (based on `Eric_cam.png`):**

* The **strongest hotspot is on the empty wall area to the left of me**, appearing as a red/orange blob.
* There is also visible activation across parts of the **flag stripes** (a yellow/green band area), and some spread on the wall around the center/right side.
* The model gives **less emphasis to my face/body** than expected, and instead highlights large background regions.

**Why this likely happened:**
The background contains strong geometric cues (thin rack lines, shadows, and long straight edges). Those shapes can resemble rigid “tool like” or “pole like” patterns in ImageNet trained features, which helps explain why labels like `flagpole` or even object like classes appear. This demonstrates how a classifier can be misled when background structures dominate the scene more than the main subject.

---

## Part 2: Creating and Experimenting with Image Filters

### 1. Understand Image Filters

**Observations on Blur Output:**
The `basic_filter.py` script uses Pillow to apply a blur (Gaussian blur). The output shows the image was **resized down and then blurred heavily**, which creates a soft, smoothed look where fine details disappear quickly. The result looks intentionally stylized because the resizing step reduces detail before the blur is applied, and then the final image is saved/displayed through Matplotlib.

---

### 2. Design Your Own Artistic Filter

**Description of Custom Filter:**
With AI assistance, I created `custom_filter.py` to generate a *Cyberpunk Neon Edge-Glow* effect. The filter pipeline:

1. Boosts **color saturation** and **contrast** to make colors more intense.
2. Detects edges using `ImageFilter.FIND_EDGES`.
3. Colorizes the edge mask into **neon cyan** (`#00FFFF`).
4. Blends the neon edges back onto the base image using **screen blending** (`ImageChops.screen`) to create a glow/outline look.

**Effect on the image:**
The final output produces sharp neon style outlines around major contours (jacket edges, facial outline, and background lines) while keeping the overall scene vivid and high contrast, creating a sci-fi/cyberpunk aesthetic.

---

## Final Reflection

**AI Collaboration Reflection:**
Collaborating with AI significantly accelerated my workflow and helped me understand libraries I’m still learning. It was especially helpful for:

* Explaining preprocessing and model input requirements for classification.
* Implementing Grad CAM correctly with TensorFlow’s `GradientTape`.
* Troubleshooting framework warnings and integration issues without breaking the pipeline.
* Turning a creative idea (“cyberpunk neon glow”) into a real filter by combining enhancement, edge detection, and blending.

Overall, the project showed how AI can function as both a debugging partner and a creative collaborator helping me go from concept → working code → interpretable results.

