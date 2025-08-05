Ayna_Internship_Assignment
This repository contains the implementation notebook and a PDF report for the Ayna Internship Assignment on polygon colorization using a Conditional UNet.

📄 Overview
The goal of this project was to design and train a model that can take a grayscale image of a polygon along with a target color name and generate a colorized output image with the polygon filled in the specified color.

We built a Conditional UNet architecture to learn this mapping, explored various conditioning methods, tuned hyperparameters, and analyzed training dynamics to arrive at an optimal setup.

🔧 Hyperparameters
What Was Tried
Learning rates: 1e-3, 5e-4, 1e-4 (✅ best)

Optimizers: Adam, AdamW (✅ Adam gave smoother convergence)

Loss weights: L1-only, MSE-only, and combination (L1: 1.0, MSE: 0.5 worked best)

Dropout values: 0.0, 0.1, 0.25

Schedulers: ReduceLROnPlateau (✅ chosen), CosineAnnealing

Final Settings
Hyperparameter	Value
Image Size	256x256
Embedding Dimension	64
Batch Size	16
Learning Rate	1e-4
Loss Function	L1 + 0.5 × MSE
Dropout	0.1
Optimizer	Adam
Scheduler	ReduceLROnPlateau
Early Stopping	15 epochs

Rationale: Final settings were chosen based on validation loss trends and visual output quality (sharpness, saturation, edge clarity).

🏗️ Architecture and Conditioning
Design Overview
Backbone: 4-level symmetric UNet with skip connections and DoubleConv modules (Conv → BN → ReLU ×2).

Conditioning: Color label is embedded via nn.Embedding (dim = 64), expanded to match spatial dimensions (64×H×W), and concatenated with 1-channel input → resulting in a 65-channel input tensor.

Encoder Path:

Initial block: 1+64 → 64

Downsampling: 64 → 128 → 256 → 512 → 1024 (via max pooling + DoubleConv)

Decoder Path: Bilinear upsampling + skip connections, followed by DoubleConv blocks.

Output Layer: 1×1 Conv → 3-channel RGB output → sigmoid activation.

Dropout: Applied within DoubleConv blocks (p=0.1).

Ablations Explored
No Conditioning: Outputs lacked diversity, always predicted a default color.

Concatenation vs. FiLM: Concatenation of embeddings as extra spatial channels gave superior results.

TransposeConv vs. Bilinear Upsampling: Both worked similarly; bilinear chosen for simplicity.

📈 Training Dynamics
Loss Curves
Training and validation losses decreased steadily. Slight overfitting appeared after epoch 60, mitigated via early stopping.

Qualitative Trends
Early epochs: Desaturated, patchy outputs.

Mid training: Colors became more vivid, shapes more accurate.

Final model: Correct colors, well-bounded polygons.

Failure Modes and Fixes
Blurry edges: Fixed by prioritizing L1 loss.

Wrong colors: Occurred when conditioning was injected too late—fixed by conditioning from the first layer.

Artifacts: Reduced using dropout and augmentation.

🧠 Key Learnings
Conditional embeddings help produce distinct outputs for identical shapes with different target colors.

UNet architecture effectively preserves polygon structure and boundaries.

Conditioning must occur early (first conv layer) to be effective.

Combining L1 and MSE losses improves visual quality.

Augmentations (rotation, flipping) enhance generalization.

✅ Conclusion
This project demonstrates that a Conditional UNet, combined with simple color label conditioning and robust training practices, can generate accurate and generalizable colorizations for polygon images.
