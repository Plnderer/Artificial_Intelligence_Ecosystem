# Image Classification and Processing Assignment Report

## Part 1: Using the Basic Classifier and Implementing Grad CAM

### 3. Run the Basic Classifier
**Image Evaluated:** `Eric.png`

**Top-3 Predictions:**
1. `chain_saw` (0.12)
2. `flagpole` (0.08)
3. `assault_rifle` (0.07)

**Reflections on Code Explanation:**
The AI provided a clear line by line breakdown of `base_classifier.py` and the Keras library dependencies. As someone focusing on understanding ML in Pythona, it was particularly illuminating to learn that models require input images to be strictly reshaped (224x224) and subsequently expanded into "batch arrays" (`np.expand_dims`) before predictions can happen. Furthermore, observing how `decode_predictions` effectively maps abstract floating point scores back into a structured, human readable format made perfect sense and clarified the pipeline.

### 4. Implement Grad CAM
Grad CAM was successfully integrated into `base_classifier.py` using TensorFlow's `GradientTape`. The implementation extracts the feature maps from the last convolutional layer (`out_relu`) and calculates the gradients of the top predicted class relative to those feature maps. It then scales these importance values into a colorized heatmap (using Matplotlib's jet colormap), superimposes it over the original image, and saves it to a new file ending in `_cam.png`.

### 5. Understand Grad CAM
Grad CAM (Gradient weighted Class Activation Mapping) is a technique that provides a visual explanation for the decisions made by CNN based models (like MobileNetV2). It works by targeting the final convolutional layer of the network, interpreting the spatial information it has learned before it's flattened for final prediction. It calculates the "gradient" (derivative) of the model's top predicted class flowing back into this layer. By averaging these gradients, it assigns an "importance weight" to each feature map, representing how much that specific map contributed to the final decision. Finally, the weighted feature maps are combined and passed through a ReLU function (to keep only positive influences) to create a visual heatmap. This heatmap is overlaid on the original image, using "hotter" colors (like red or yellow) to highlight the specific areas that most strongly influenced the model to make its prediction.

### 6. Analyze the Heatmap
For the image `Eric.png`, the model's top prediction was `chain_saw` with a very low confidence score of 12%, followed by `flagpole` and `assault_rifle`. The original image shows a person standing indoors beneath a Puerto Rican flag. However, looking at the generated Grad-CAM heatmap, the model completely ignores both the person and the flag. The strongest highlights (the red, orange, and yellow areas) are located in the empty space on the wall to the left of the person and slightly below the flag, near the shadow and the vertical rack on the wall. The model seems to have fixated on these vertical background structures, likely confusing the dark vertical lines and shadows for objects like a `flagpole` or parts of a `chain_saw` or `assault_rifle`. This demonstrates a case where the model failed to identify the main subjects and was misled by background elements, resulting in poor predictions.


## Part 2: Creating and Experimenting with Image Filters

### 1. Understand Image Filters
**Observations on Blur Output:**
The `basic_filter.py` script relies on the traditional Python Imaging Library (Pillow). Inspecting the script revealed that it forces the image down into a smaller 128x128 coordinate space before executing an `ImageFilter.GaussianBlur(radius=2)` mechanism. This sequential downsizing and blurring translates to a heavily stylized, softened, and mildly pixelated graphic. The use of the `matplotlib.pyplot` module to physically display and export the augmented asset instead of Pillow's native mechanisms was an interesting creative decision.

### 2. Design Your Own Artistic Filter
**Description of Custom Filter:**
With AI assistance, I conceptualized and developed a new script known as `custom_filter.py` to impart a *Cyberpunk Neon Edge-Glow* style. The custom program works sequentially by maximizing the original color saturation and contrast (`ImageEnhance`). Then, it isolates all hard contours within the subject using an `ImageFilter.FIND_EDGES` mask. These outlines are forcibly colorized into a shocking, neon cyan (`#00FFFF`) and then overlayed back onto the base subject image via a screen-blending operation (`ImageChops.screen`). The final product boasts a vibrant, glowing sci-fi aesthetic.


## Final Reflection
**AI Collaboration Reflection:**
Collaborating with AI significantly accelerated my workflow and provided deep context to otherwise complex libraries. Among the predominant challenges was navigating framework-specific deprecation warnings originating from TensorFlow and Keras configurations when integrating the advanced Grad-CAM methodology; fortunately, the AI seamlessly patched these with precise imports without breaking the pipeline. Expanding on my successes, the AI truly shined during the custom filter design constraint. While I simply iterated a high-level creative vision for a "cyberpunk neon line" look, the AI mathematically decoded that concept into distinct computational layers (leveraging ImageOps maps and Screen logic) that drastically exceeded the capabilities of a beginner programmer. This project highlighted how AI functions exceptionally well as both a meticulous debugging tool and an inventive creative partner.
