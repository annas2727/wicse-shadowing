# Gameplay Audio Classifier (CNN)

This project aims to build a Convolutional Neural Network (CNN) to identify various sounds within gameplay audio clips, specifically designed for multi-label classification (meaning a single audio segment can have multiple sounds present).

## Project Structure

-   `cnnStuff/`: Contains the core Python package for the CNN model, data collection, training, and prediction scripts.
    -   `cnnStuff/src/cnnstuff/`: Python source code.
        -   `audio_model.py`: Defines the CNN architecture.
        -   `collect_data.py`: Script for model-assisted labeling of new audio chunks.
        -   `train_model.py`: Script for training the CNN model.
        -   `predict.py`: Script for making predictions on individual audio files.
        -   `evaluate.py`: Script for interactively evaluating and correcting existing labels.
-   `data/`: This directory is crucial for all data-related assets.
    -   `data/labels.json`: Defines the list of all possible sound labels.
    -   `data/manual_labels.json`: Stores your human-curated labels for audio chunks.
    -   `data/audio_chunks/`: Contains the 3-second WAV audio chunks extracted from your source audio/video files.
    -   `data/skipped_files.json`: (Optional) Stores names of chunks you explicitly skipped during initial labeling.
    -   `audio_model.pth`: (Generated after training) The saved weights of your trained CNN model.
-   `gameplay_720p.mp4`, `counter_strike_audio.m4a`, `counter_strike_audio.wav`: Your raw audio/video files (these should be moved into the `data/` directory if not already).

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry:** If you don't have Poetry installed, follow the instructions on their official website.
2.  **Navigate to `cnnStuff/`:** Open your terminal and change directory into the `cnnStuff/` folder (where `pyproject.toml` is located).
    ```bash
    cd cnnStuff/
    ```
3.  **Install Dependencies:** Install all project dependencies and create the virtual environment.
    ```bash
    poetry install
    ```
4.  **Activate Shell (Optional but Recommended):** To work within the Poetry environment, you can activate its shell.
    ```bash
    poetry shell
    ```
    (You can exit the shell with `exit` when done.)

## Workflow

The typical workflow involves collecting data, training the model, and then iteratively evaluating and improving your labels.

### 1. Define Your Labels (`data/labels.json`)

Before anything else, ensure `data/labels.json` contains all the sound categories you want your model to identify. This file is a simple JSON array of strings.

Example `data/labels.json`:
```json
[
    "footsteps",
    "gunshot",
    "gun_handling",
    "explosion",
    "knife",
    "interface",
    "background"
]
```

### 2. Collect and Label New Data (Model-Assisted)

Use `collect_data.py` to process new audio/video files, split them into chunks, and label them with assistance from your trained model.

-   Place your new audio/video file (e.g., `new_gameplay.mp4`) into the `data/` directory.
-   Run the script from the `cnnStuff/` directory:
    ```bash
    poetry run python -m cnnstuff.collect_data ../data/new_gameplay.mp4
    ```
-   The script will:
    -   Split the file into 3-second chunks and save them in `data/audio_chunks/`.
    -   Load your trained model (`audio_model.pth`).
    -   For each *unlabeled* chunk:
        -   Play the audio automatically.
        -   Show the model's prediction (including confidence scores for all labels).
        -   Prompt you to `Accept (y)`, `Correct (n)`, `Replay (r)`, or `Quit (q)`.
        -   If you choose `n`, you can enter multiple correct labels separated by commas (e.g., `1,3`).
-   All new and corrected labels will be saved to `data/manual_labels.json`.

### 3. Train the Model

After collecting and labeling a sufficient amount of data, train your CNN model.

-   Run the script from the `cnnStuff/` directory:
    ```bash
    poetry run python -m cnnstuff.train_model
    ```
-   This will train the `AudioCNN` model using the labels in `data/manual_labels.json` and save the trained model weights to `audio_model.pth` in the project root.

### 4. Evaluate and Refine Existing Labels

Use `evaluate.py` to review the quality of your existing labels and correct any inconsistencies or errors. This is crucial for improving model performance.

-   Run the script from the `cnnStuff/` directory:
    ```bash
    poetry run python -m cnnstuff.evaluate
    ```
-   The script will:
    -   Iterate through all chunks in `data/manual_labels.json`.
    -   Play the audio automatically.
    -   Show your "Ground Truth" label (from `manual_labels.json`) and the model's prediction.
    -   Prompt you to confirm if the prediction is `Correct (y)`, `Incorrect (n)`, `Replay (r)`, or `Quit (q)`.
    -   If you choose `n`, you can provide corrected labels (comma-separated).
-   Any corrections you make will update `data/manual_labels.json`. After making corrections, you should retrain the model (step 3).

### 5. Make Predictions on Individual Files

Use `predict.py` to get a detailed prediction for a single audio chunk.

-   Run the script from the `cnnStuff/` directory:
    ```bash
    poetry run python -m cnnstuff.predict ../data/audio_chunks/your_audio_chunk.wav
    ```
-   This will output the model's confidence for each label and its final prediction based on the set threshold.

---

This iterative process of labeling new data with model assistance, training, and then evaluating/refining your labels is key to building a robust audio classification system.
