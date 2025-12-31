# Project-2 Specification

**Project Overview**

Banks, universities, e-governance systems and companies still rely heavily on **offline handwritten signatures** for:

* Cheque processing
* KYC forms
* Agreement signing
* Exam answer sheets, certificates, ID cards

AI-based **signature verification systems** (SVS) need lots of samples per person to learn:

* Natural variation in genuine signatures
* Differences between genuine and forged signatures

But in practice:

* Each user gives only a **few genuine samples**
* Forgeries are rare (or simulated)
* Collecting extra genuine samples is inconvenient

A **Vanilla GAN** can learn the **distribution of handwritten signatures** and generate new synthetic variants that:

* Looks like genuine signatures of a person (or style)
* Increase the diversity of genuine samples
* Help train more robust verification models

Project Goal: Design and implement a **Vanilla GAN** to:

1. Learn the distribution of real handwritten signatures (person-specific or generic).
2. Generate realistic synthetic signature images (unconditional).
3. Use synthetic signatures to **augment training data** for a signature verification model.
4. Evaluate impact on **verification accuracy, FAR, FRR**.
5. Provide a simple **tool/UI** for generating synthetic signature datasets.

**Real-Life Use Cases**

* Banks/financial institutions are improving cheque/mandate verification.
* Universities verifying exam/record signatures.
* E-governance services (paper forms, certificates).
* R&D labs working on offline signature verification.

**2. High-Level System Architecture**

**![](data:image/png;base64...)**

**Module-Wise Specification**

**Module 1 — Data Pipeline & Preprocessing**

**Objective:** Prepare a clean set of signature images for GAN training.

**Data Sources**

* Public signature datasets (e.g., GPDS, CEDAR, MCYT) or
* Institution’s internal scanned signatures (with proper permissions).

**Scope Choice:** Two options:

* **Person-specific model**
  + Train one Vanilla GAN per person (small scale).
* **Generic signature model**
  + Train on many users; outputs “generic” realistic signatures.

For a first project, use a **generic** or a **small subset (5–10 users)**.

**Preprocessing Steps:**

* Convert scans to grayscale.
* Binarize or normalize intensity (depending on quality).
* Crop around signature (remove margins).
* Resize images to a fixed size (e.g., 64×64 or 128×128).
* Normalize pixel values to [−1, 1] (for tanh output).

Optional:

* Centring and aspect-ratio normalisation.
* Remove extreme noise / badly scanned samples.

**Deliverables:**

* data\_loader\_signatures.py
* preprocess\_signatures.py
* Directory: data/signatures/train/ with pre-processed PNGs/JPEGs

**Module 2 — Vanilla GAN Architecture**

**Objective:** Simple GAN: **unconditional**, no labels, only real vs fake.

**Generator (G)**

* Input: latent vector z ∈ R^100 sampled from N(0, I).
* Architecture (for 64×64 images):
  + Dense → reshape to small feature map (e.g., 4×4×256).
  + Sequence of upsampling blocks:
    - ConvTranspose2D (or Upsample+Conv2D)
    - BatchNorm
    - ReLU activation
  + Repeat until output size (64×64×1).
  + Final layer:
    - Conv2D with 1 channel
    - tanh activation (to match [−1,1] input normalization).

Output: synthetic signature image.

**Discriminator (D)**

* Input: 64×64×1 image (real or fake).
* Architecture:
  + Conv2D + LeakyReLU(0.2) + Dropout (optional).
  + Downsampling (stride 2 or maxpool).
  + Repeat until small feature map.
  + Flatten + Dense(1) with sigmoid.

Output: scalar probability of being “real”.

**Loss Functions (Classical Vanilla GAN)**

* Use **binary cross-entropy (BCE)**:
  + Discriminator loss:
    - L\_D = -[log(D(x\_real)) + log(1 - D(G(z)))]
  + Generator loss:
    - L\_G = -log(D(G(z)))

Alternative: label smoothing (e.g., real label = 0.9) to stabilize D.

**Optimizer & Hyperparameters**

* Optimizer: Adam
  + lr = 2e-4
  + β1 = 0.5, β2 = 0.999
* Batch size: 64
* Epochs: 100–200 (depending on dataset size)

**Deliverables:**

* generator\_vanilla\_gan.py
* discriminator\_vanilla\_gan.py
* vanilla\_gan\_model.py

**Module 3 — Training Engine**

**Objective:** Implement training loop, logging, and image saving.

**Training Loop**

For each epoch:

**Train Discriminator:**

* + Sample a batch of real signatures x\_real.
  + Sample a batch of noise z → generate x\_fake = G(z).
  + Compute:
    - D\_real\_loss = BCE(D(x\_real), 1)
    - D\_fake\_loss = BCE(D(x\_fake\_detached), 0)
  + Total L\_D = D\_real\_loss + D\_fake\_loss.
  + Update D parameters.

**Train Generator:**

* + Sample z → x\_fake = G(z).
  + Generator wants D(x\_fake) ≈ 1:
    - G\_loss = BCE(D(x\_fake), 1).
  + Update G parameters.

**Logging:**

* + Record G\_loss, D\_loss per batch/epoch.
  + Every N epochs:
    - Save sample grid of generated signatures.

**Stabilization Techniques**

* Label smoothing (e.g., real = 0.9).
* Small Gaussian noise on real inputs to D.
* Clip gradients if exploding.
* Monitor for **mode collapse** (all signatures look identical).

**Checkpointing**

* Save G\_epoch\_XXX.pth/.h5 and D\_epoch\_XXX.pth/.h5.
* Save “fixed noise” samples so progress can be visually compared.

**Deliverables:**

* train\_vanilla\_gan\_signatures.py
* checkpoints/
* samples/ (generated signatures during training)
* logs/ for plotting curves

**Module 4 — Evaluation & Performance Assessment**

**4A. Visual Inspection**

* Generate a large batch (e.g., 500–1000) of signatures.
* Inspect:
  + Stroke smoothness
  + Pen thickness consistency
  + No bizarre artefacts (blocks, noise).
* Visual grid by random sampling.

If using **person-specific** GAN:

* Compare synthetic signatures visually with genuine ones for that person.

**4B. Image-Level Metrics & Diversity**

Even though signatures are tricky for standard metrics, you can still use:

* **FID** (Fréchet Inception Distance) on resized, replicated channels.
* **LPIPS** diversity (different z → different images).
* Simple statistics:
  + Stroke density distribution
  + Foreground pixel count per image
  + Aspect ratio & orientation variations

Goal: synthetic distribution should match real distribution reasonably.

**4C. Verification Model Impact (Core Experiment)**

This is the key “real-life” part.

1. **Train a baseline signature verification model:**
   * Siamese CNN, or simple CNN classifier for genuine vs forgery (depending on dataset).
   * Training only with **real genuine signatures** (+ available forgeries).
2. **Train an augmented verification model:**
   * Use **real genuine + synthetic genuine** signatures for training.
   * Forgeries: real ones, or random mismatched-user signatures.
3. **Evaluation:**
   * On **real test set**, compute:
     + Accuracy
     + False Acceptance Rate (FAR)
     + False Rejection Rate (FRR)
     + Equal Error Rate (EER)

Expected outcome:

* Lower EER / FAR / FRR when synthetic signatures are included (especially if genuine data was scarce).

**Deliverables:**

* signature\_verifier\_train.py
* signature\_verifier\_eval.py
* Tables/plots comparing baseline vs augmented models
* ROC curves, DET curves

**4D. Ablation Studies (Optional)**

* Compare:
  + Vanilla GAN vs DCGAN (same dataset).
  + Different latent dimensions (50, 100, 200).
  + Different activation choices (ReLU vs LeakyReLU).

Metrics: training stability, visual quality, verification impact.

**Deliverables:**

* ablation\_vanilla\_gan\_signatures.py
* Summary tables/graphs

**Module 5 — Deployment: Synthetic Signature Tool**

**Objective:** Provide a simple tool for generating synthetic signature datasets.

**Inference Script**

* generate\_signatures.py:
  + Inputs:
    - Number of signatures N
    - Output directory
    - (Optional) random seed
  + Loads: G\_final.pth
  + Outputs: N synthetic signature images.

If person-specific models exist:

* Pass user\_id to choose correct model or model checkpoint.

**UI (Simple Web App — Streamlit/Gradio)**

Features:

* Input:
  + Number of signatures to generate
  + (Optional) user/person ID
* Button: “Generate Signatures”
* Display:
  + Gallery of generated signatures
* Download button for ZIP

**(Optional) REST API**

* POST /generate:
  + { "n": 100, "user\_id": "A123" }
* Returns: ZIP or list of images encoded/base64.

**Deliverables:**

* app\_vanilla\_gan\_signatures.py
* api\_vanilla\_gan\_signatures.py

**Module 6 — Monitoring, Versioning & Future Work**

**Objective:** Treat the GAN as part of a verification system pipeline.

**Model Versioning**

* VanillaGAN-Sig-v1.0: baseline for generic signatures.
* v1.1: improved architecture or higher resolution.
* v2.0: person-specific models, more users.

Store metadata:

* Dataset used
* Training settings
* Verification impact metrics

**Monitoring Usage**

If integrated into an SVS pipeline

* Log when/where synthetic data is used.
* Track model performance after retraining with synthetic data.

**Future Extensions**

* Conditional GAN per user (CGAN/ACGAN) for more control.
* Higher resolution signatures (e.g., 256×256).
* Style transfer between signatures (StyleGAN or encoder+GAN).

**Deliverables:**

* model\_versions.yaml
* future\_work.md

**Deliverables Summary**

**Documentation**

* Project proposal / SRS
* System architecture & design doc
* Vanilla GAN architecture description
* Experimental & evaluation report
* User/deployment manual

**Code Skeleton (Example)**

vanilla\_gan\_signatures/

├── src/

│ ├── data\_loader\_signatures.py

│ ├── preprocess\_signatures.py

│ ├── generator\_vanilla\_gan.py

│ ├── discriminator\_vanilla\_gan.py

│ ├── vanilla\_gan\_model.py

│ ├── train\_vanilla\_gan\_signatures.py

│ ├── evaluate\_vanilla\_gan\_signatures.py

│ ├── signature\_verifier\_train.py

│ ├── signature\_verifier\_eval.py

│ ├── ablation\_vanilla\_gan\_signatures.py

│ ├── generate\_signatures.py

│ ├── app\_vanilla\_gan\_signatures.py

│ ├── api\_vanilla\_gan\_signatures.py

│ └── utils/

│ ├── metrics.py

│ ├── logger.py

│ └── visualizer.py

├── data/

├── checkpoints/

├── samples/

├── figures/

├── logs/

└── docs/
