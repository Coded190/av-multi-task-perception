# AV Perception Project: End-to-End Multi-Task Learning Roadmap

This master document outlines the implementation strategy, file architecture, and phased development steps for building a lightweight, custom Autonomous Vehicle (AV) perception model from scratch using PyTorch. 

---

## 1. Core Architectural Strategy
* **Task Configuration:** Multi-Task Learning (MTL) system combining 2D Object Detection (bounding boxes for vehicles/pedestrians) and Semantic Segmentation (pixel mask for drivable area/lanes).
* **Hardware Constraint:** Optimized for a **16 GB VRAM GPU** via Automatic Mixed Precision (AMP), input resolution matching ($640 \times 640$), and efficient backbone design.
* **Engineering Principles Applied:** * **Separation of Concerns (SoC):** Distinct separation between data processing, neural routing, penalty definitions, and hardware execution.
  * **Command-Query Separation (CQS):** Training state mutation (`trainer.py`) is completely decoupled from read-only validation telemetry (`evaluator.py`).
  * **SOLID / Open-Closed:** Components (like backbones) can be swapped out without altering downstream heads or execution loops.

---

## 2. Exhaustive Directory & File Blueprint

### Configuration & Data Layer
* **`config/base_config.yaml`**
  * *Purpose:* Single source of truth (KISS/YAGNI). Stores hyperparameters (learning rate, weight decay, loss multipliers), data directories, resolution (`imgsz: 640`), batch size, and device configuration (`cuda`).
* **`data/`**
  * *`raw/`:* Immutable raw video clips or original dashcam sequences.
  * *`processed/`:* Extracted image frames downscaled to target working resolutions.
  * *`annotations/`:* Labels stored in a unified JSON (COCO-style) or text layout map containing both bounding box arrays and polygon coordinates.

### Data Pipeline Module (`src/datasets/`)
* **`src/datasets/av_dataset.py`**
  * *Purpose:* Custom PyTorch class extending `torch.utils.data.Dataset`. Implements `__len__` and `__getitem__`. Responsible for loading images via OpenCV/PIL, parsing labels, and packing targets into standard dictionaries containing `boxes`, `labels`, and `masks`.
* **`src/datasets/transforms.py`**
  * *Purpose:* Wraps data augmentation software (e.g., Albumentations). Exposes structural policies for train/val data transformations (e.g., horizontal flips, color jitter) ensuring bounding boxes and polygon masks remain structurally synchronized with spatial alterations.

### Model Architecture Module (`src/models/`)
* **`src/models/backbone.py`**
  * *Purpose:* Inherits from `nn.Module`. Functions as the deep visual feature extractor. You can write custom convolutional block layers or load a lightweight feature extractor (like a small ResNet or MobileNet), cutting it off before the final classification layer.
* **`src/models/neck.py`**
  * *Purpose:* Implements feature aggregation (e.g., Feature Pyramid Network / FPN). Receives low-level, high-resolution feature maps from early layers and merges them with high-level, high-semantic maps from deep layers. Necessary to handle multi-scale objects (e.g., close-range cars vs. distant pedestrians).
* **`src/models/heads.py`**
  * *Purpose:* Houses two distinct specialized sub-networks:
    1. `DetectionHead`: Linear or 1x1 Conv layers mapping neck features to class scores and box boundaries $(x, y, w, h)$.
    2. `SegmentationHead`: Deconvolutional / Bilinear Upsampling layers scaling downsized feature grids back up to input dimension maps $(B, \text{Classes}, 640, 640)$.
* **`src/models/multi_task_net.py`**
  * *Purpose:* Orchestrator module. Acts as the master `nn.Module` wrapper that connects everything together: feeds input tensor to `backbone`, passes outputs to `neck`, and routes aggregated layers to both `heads` simultaneously, returning a dictionary containing detection and segmentation outputs.

### Loss Definitions (`src/losses/`)
* **`src/losses/detection_loss.py`**
  * *Purpose:* Computes object localization errors (using a bounding-box metric like CIoU Loss) alongside object category classification penalties (using Cross-Entropy).
* **`src/losses/segmentation_loss.py`**
  * *Purpose:* Measures accuracy of the pixel mask predictions. Combines Binary Cross-Entropy (BCE) with Dice Loss to effectively handle class imbalances (e.g., when the road occupies a large part of the image, but lane lines are thin).
* **`src/losses/total_loss.py`**
  * *Purpose:* Aggregates specific losses into a singular gradient-scalable scalar. Implements task balancing: $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{det}} + \beta \mathcal{L}_{\text{seg}}$, where $\alpha, \beta$ are weight parameters tuned from config variables.

### Execution Engine Layer (`src/engine/`)
* **`src/engine/trainer.py`**
  * *Purpose:* Command execution. Owns mutable state (optimizer, learning rate schedulers, model parameter maps). Orchestrates the forward pass, loss derivation, backpropagation, and state updates using modern PyTorch Mixed Precision (`torch.amp.autocast`).
* **`src/engine/evaluator.py`**
  * *Purpose:* Query execution. Read-only validation harness. Locks the model in `.eval()` state with gradients disabled (`torch.no_grad()`). Aggregates validation predictions across the dataset to compute global validation metrics.

### Utility & Core Logic Module (`src/utils/`)
* **`src/utils/metrics.py`**
  * *Purpose:* Stateless, mathematical library containing pure functions (DRY). Evaluates detection quality via Mean Average Precision (mAP@0.5 and mAP@0.5:0.95) and checks mask accuracy via Intersection-over-Union (IoU).
* **`src/utils/visualization.py`**
  * *Purpose:* Interface layer for displaying outputs. Draws bounding boxes with class text tags and blends semi-transparent colored masks over processed frames using OpenCV.

### System Entry Points (Root Scripts)
* **`train.py`**
  * *Purpose:* Script to run the complete training pipeline. Parses `config/base_config.yaml`, instantiates data objects, constructs your network, runs the training loop through `src/engine/trainer.py`, and serializes the final weights (`.pth`).
* **`eval.py`**
  * *Purpose:* Standalone validation program. Loads saved weights, feeds the test dataset into `src/engine/evaluator.py`, and generates a final report detailing precision metrics.
* **`infer.py`**
  * *Purpose:* Production/Inference execution. Sets up a loop to pull frames from a dashcam feed or raw video file, routes them through the trained model, applies Non-Maximum Suppression (NMS) to clear overlapping boxes, and renders a live, visual overlay.

---

## 3. Phased Step-by-Step Implementation Roadmap

Follow these phases sequentially. Do not move to the next phase until the current phase's outputs can be programmatically verified.

### Phase 1: Data Preparation & Frame Extraction
1. Gather target driving clips and place them into `data/raw/`.
2. Construct a utility script to sample frames periodically (e.g., every 5th frame) to ensure diversity. Write them to `data/processed/`.
3. Load frames into an annotation tool (e.g., CVAT or Label Studio). Define bounding boxes for objects and polygon sequences for drivable surfaces or lanes. Export them to `data/annotations/`.

### Phase 2: Configuration & Data Ingestion Pipeline
1. Fill out your `config/base_config.yaml` parameters.
2. Code `src/datasets/transforms.py` to handle simple tensor conversion and resizing to $640 \times 640$.
3. Implement `src/datasets/av_dataset.py`. Confirm your `__getitem__` logic works correctly by writing a temporary script to extract a single batch and manually verifying that tensor shapes are structured properly.

### Phase 3: Building the Custom Multi-Task Network
1. Open `src/models/backbone.py` and implement a modular feature extractor using basic PyTorch layers, or strip the head off a pre-trained model.
2. Code `src/models/neck.py` to route and combine spatial feature layers.
3. Build the specific task layers inside `src/models/heads.py`.
4. Tie the three components together inside `src/models/multi_task_net.py`. Verify it by running a dummy image tensor through the model to confirm it correctly outputs separate detection and segmentation tensors without error.

### Phase 4: Loss System Formulation
1. Write the detection criteria in `src/losses/detection_loss.py`.
2. Write the mask scoring criteria in `src/losses/segmentation_loss.py`.
3. Set up the combined tracking logic in `src/losses/total_loss.py`.

### Phase 5: Writing the Core Execution Engines
1. Implement the evaluation metrics inside `src/utils/metrics.py`.
2. Code `src/engine/evaluator.py` to loop over validation tensors and return tracking metrics.
3. Build the core `src/engine/trainer.py` framework. Add modern `torch.amp` context blocks here to enable 16 GB VRAM optimization.

### Phase 6: Operational Assembly & Training Run
1. Wire all completed parts together into your master `train.py` script.
2. Run an initial check using a tiny batch size (e.g., 2 images) to verify the loss decreases and confirm the pipeline functions end-to-end without breaking.
3. Launch a full training session. Use `torch.compile(model)` if your setup supports it to maximize training performance.

### Phase 7: Evaluation & Verification
1. Build `eval.py` to verify the accuracy of your saved weights against the test dataset.
2. Check your mAP and IoU values to ensure the model isn't underfitting or overfitting.

### Phase 8: Real-Time Visualization Pipeline
1. Program `src/utils/visualization.py` to map bounding boxes and colored masks onto video frames.
2. Write the final `infer.py` file to stream video inputs, apply prediction logic, run Non-Maximum Suppression (NMS) to clean up duplicate boxes, and display the final visual output in real-time.

---

## 4. VRAM Checklists (16 GB Guardrails)
If you hit Out-Of-Memory (OOM) errors during Phase 6, check these settings:
* [ ] Is input image resolution strictly limited to $640 \times 640$ or $512 \times 512$?
* [ ] Is `torch.amp.autocast('cuda')` actively wrapping forward operations in `trainer.py`?
* [ ] Is the data loader configured to run with `pin_memory=True`?
* [ ] If needed, have you lowered the batch size in `base_config.yaml` and implemented gradient accumulation steps in your trainer?
* [ ] Are you calling `optimizer.zero_grad(set_to_none=True)` instead of standard `zero_grad()` to save memory on your GPU?

## Exhaustive Directory & File Blueprint

### Visual Project Tree
```text
av_perception/
├── config/
│   └── base_config.yaml       # KISS/YAGNI: Single source of truth for hyperparameters, paths, and resolutions
├── data/                      # Separated entirely from source code (SoC)
│   ├── raw/                   # Immutable original images and video clips
│   ├── processed/             # Resized or formatted frames
│   └── annotations/           # JSON/YOLO labels for boxes and polygon masks
├── src/                       # Source directory containing all core modules
│   ├── __init__.py
│   ├── datasets/              # Responsibility: Data parsing, transformations, and batching
│   │   ├── __init__.py
│   │   ├── av_dataset.py      # Custom PyTorch Dataset class
│   │   └── transforms.py      # Image augmentations wrapper (Albumentations)
│   ├── models/                # Responsibility: Structural definition of the network (nn.Module)
│   │   ├── __init__.py
│   │   ├── backbone.py        # Feature extractor (e.g., custom or MobileNet/ResNet)
│   │   ├── neck.py            # FPN / Feature Aggregation Layer
│   │   ├── heads.py           # Multi-task splitting: DetectionHead and SegmentationHead
│   │   └── multi_task_net.py  # Orchestrates backbone -> neck -> heads
│   ├── losses/                # Responsibility: Math & penalties (SOLID: Single Responsibility)
│   │   ├── __init__.py
│   │   ├── detection_loss.py  # Box and class loss functions
│   │   ├── segmentation_loss.py # Dice / BCE loss for lanes/drivable space
│   │   └── total_loss.py      # Aggregates and weights the individual heads
│   ├── engine/                # Responsibility: Execution state (CQS: Command-Query Separation)
│   │   ├── __init__.py
│   │   ├── trainer.py         # COMMAND: Modifies model state (backprop, AMP, optimization)
│   │   └── evaluator.py       # QUERY: Inspects performance (calculates mAP and IoU without side effects)
│   └── utils/                 # Responsibility: Shared stateless helper operations (DRY)
│       ├── __init__.py
│       ├── metrics.py         # Pure functions for mAP, IoU calculations
│       └── visualization.py   # Pure functions for drawing bounding boxes and masks on video
├── train.py                   # Entry point for starting the training run
├── eval.py                    # Entry point for validating model against test data
├── infer.py                   # Entry point for real-time video/live-camera inference
├── requirements.txt
└── README.md
