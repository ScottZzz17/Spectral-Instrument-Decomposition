# AI-Driven Music Deconstruction Project

A personal project that combines AI (deep learning) and music signal processing techniques to deconstruct a song into its fundamental components. The system processes an audio file by interpreting each beat as a waveform, analyzing wavelength density to determine instrument count per beat, and identifying individual instruments and notes. The final output is designed for export into a Digital Audio Workstation (DAW) for visual reconstruction of the song’s creation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Core Objectives](#core-objectives)
- [Design & Problem-Solving Process](#design--problem-solving-process)
  - [Design Approach](#design-approach)
  - [Challenges & Solutions](#challenges--solutions)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Usage Guidelines](#usage-guidelines)
- [Demonstrative Code](#demonstrative-code)
  - [Beat Detection & Spectral Analysis](#beat-detection--spectral-analysis)
  - [Spectral Analysis & Wavelength Density](#spectral-analysis--wavelength-density)
  - [Frequency Separation](#frequency-separation)
- [Development Roadmap](#development-roadmap)
- [Resources](#resources)
- [Progress](#progress)
- [Notes](#notes)
- [Contact Information](#contact-information)

---

## Project Overview

This project leverages advanced deep learning and music signal processing to break down a song into its core elements. By segmenting an audio file into beats and analyzing the spectral (wavelength) density of each beat, the system estimates the number of instruments playing and identifies specific instruments and notes. The final output can be exported into a DAW, providing a visual and editable reconstruction of the song.

---

## Core Objectives

- **Audio Data Acquisition:**  
  Collect audio files with corresponding instruments and note annotations from public datasets or via custom annotations.

- **Audio Preprocessing:**  
  Precisely detect beats and convert each into a spectral representation using Fourier or wavelet transform techniques.

- **Model Development:**  
  Train machine learning models to classify instrument presence and detect individual notes for every beat.

- **DAW Integration:**  
  Generate DAW-compatible outputs (e.g., MIDI files) that reconstruct the song’s structure for further visualization and editing.

---

## Design & Problem-Solving Process

### Design Approach

- **Modular Architecture:**  
  The system is built in independent modules:
  - **Data Acquisition:** Collect and annotate audio.
  - **Preprocessing:** Detect beats and extract spectral features.
  - **Modeling:** Train models to analyze each beat.
  - **Integration:** Convert predictions into DAW-friendly formats.

- **Algorithm & Method Selection:**  
  - Leverage established libraries (e.g., Librosa) for beat detection.
  - Utilize Fourier and wavelet transforms for detailed spectral analysis.
  - Experiment with different machine learning models (CNN, RNN, Transformers) to balance accuracy and performance.

- **Validation Strategy:**  
  Compare reconstructed outputs with the original track by importing them into various DAWs. Continuous testing ensures the system reliably captures the musical nuances.

### Challenges & Solutions

- **Challenge 1: Accurate Beat Detection**  
  *Problem:* Inconsistent beat tracking can lead to misaligned segments.  
  *Solution:* Fine-tuned Librosa's parameters through iterative testing on diverse audio clips to achieve precise beat segmentation.

- **Challenge 2: Capturing Detailed Spectral Features**  
  *Problem:* Subtle variations in frequency content can be lost with standard transforms.  
  *Solution:* Experimented with both Fourier and wavelet transforms, adjusting window sizes and overlaps to optimize the representation of each beat.

- **Challenge 3: Differentiating Overlapping Instrument Sounds**  
  *Problem:* Overlapping sounds and noise complicate instrument classification.  
  *Solution:* Began with classical models as baselines, then enhanced deep learning architectures with cross-validation and hyperparameter tuning to improve accuracy.

- **Challenge 4: Converting Predictions into DAW-Compatible Output**  
  *Problem:* Ensuring the mapping from model outputs to MIDI events preserves musical detail.  
  *Solution:* Developed a custom converter and tested extensively across multiple DAWs (e.g., Ableton Live, FL Studio, LMMS) to refine the translation process.

---

## Key Features

- **End-to-End Workflow:**  
  Seamless progression from audio data collection to DAW output generation.

- **Advanced Signal Processing:**  
  Employs state-of-the-art techniques to extract and analyze spectral features of each beat.

- **Modular and Scalable Design:**  
  Independent modules facilitate iterative improvements and scalability.

- **Rigorous Testing & Validation:**  
  Continuous integration and extensive testing ensure high fidelity in reconstructing musical elements.

---

## System Architecture

- **Data Processing Module:**  
  Ingests and standardizes audio data while safeguarding proprietary methods.

- **Predictive Modeling Engine:**  
  Houses machine learning models that analyze spectral data; details are kept confidential.

- **Integration Layer:**  
  Converts model outputs into DAW-friendly formats, such as MIDI files, with robust error handling.

- **Monitoring & Logging:**  
  Comprehensive logging tracks system performance and assists in troubleshooting and iterative development.

---

## Usage Guidelines

This repository serves as a technical showcase for recruiters and collaborators. The complete codebase is private to protect intellectual property. This documentation outlines the system design, methodologies, and sample implementations.

- **Review Only:**  
  This repository is for demonstration purposes; proprietary code is not downloadable.

- **Demonstrative Access:**  
  Please contact me for live demos or detailed discussions to arrange access under a non-disclosure agreement.

- **Feedback & Collaboration:**  
  Interested parties are encouraged to reach out to discuss technical details and potential partnerships.

---

## Demonstrative Code

### Beat Detection & Spectral Analysis

This snippet demonstrates beat detection using Librosa and a basic approach to spectral analysis.

```python
import librosa
import numpy as np
import pandas as pd

def detect_beats(audio_file, sr=22050):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=sr)
    # Detect beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return beat_times

def spectral_analysis(audio_file, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    # Compute Short-Time Fourier Transform (STFT)
    S = np.abs(librosa.stft(y))
    # Calculate average spectral energy per frame (as a proxy for wavelength density)
    density = np.mean(S, axis=0)
    return density

if __name__ == '__main__':
    audio_file = 'data/audio_sample.wav'
    beats = detect_beats(audio_file)
    density = spectral_analysis(audio_file)
    
    # Create a simple DataFrame to display results
    df = pd.DataFrame({'BeatTime': beats, 'SpectralDensity': density[:len(beats)]})
    print(df.head())
```

---

## Spectral Analysis & Wavelength Density

```python
import librosa
import numpy as np

def analyze_spectral_density(audio_file, sr=22050):
    """
    Analyze the spectral density of an audio file:
      - Compute the Short-Time Fourier Transform (STFT).
      - Calculate the average spectral energy per frame as a proxy for wavelength density.
    """
    y, sr = librosa.load(audio_file, sr=sr)
    S = np.abs(librosa.stft(y))
    density = np.mean(S, axis=0)
    return density

if __name__ == '__main__':
    density = analyze_spectral_density('data/audio_sample.wav')
    print("Spectral density per frame:", density)
```

---

## Frequency Separation

```python
import librosa
import numpy as np

def separate_frequencies(audio_file, sr=22050, threshold=1000):
    """
    Separate the spectral components of an audio file:
      - Compute the STFT.
      - Divide the spectrum into low and high frequencies based on a threshold.
    """
    y, sr = librosa.load(audio_file, sr=sr)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    low_mask = freqs < threshold
    high_mask = freqs >= threshold
    
    low_freq_component = S[low_mask, :]
    high_freq_component = S[high_mask, :]
    
    return low_freq_component, high_freq_component

if __name__ == '__main__':
    low, high = separate_frequencies('data/audio_sample.wav')
    print("Low frequency component shape:", low.shape)
    print("High frequency component shape:", high.shape)
```

---

## Development Roadmap

### Current Progress:
- **Data Pipeline:**  
  Implemented audio ingestion, cleaning, and initial spectral analysis; beat detection is verified.
- **Model Training:**  
  Early prototypes of the deep learning model are under active testing. The model is currently in the training and debugging phase, with challenges in data preprocessing and convergence.
- **Integration:**  
  Initial DAW output conversion (e.g., MIDI) is in beta testing.
- **Monitoring & Logging:**  
  A robust logging framework is in place to capture performance metrics and guide iterative improvements.

### Future Enhancements:
- Complete integration tests across all modules.
- Optimize model performance with larger, diverse datasets.
- Transition securely from testing/paper trading to live production following rigorous risk assessments.
- Enhance system modularity through containerization for scalability.

---

## Resources

- **Data Sources:**
  - MedleyDB
  - MusicNet
  - MIR-1K
- **Audio Software:**
  - Ableton Live (commercial)
  - LMMS (open source)
- **ML Frameworks:**
  - TensorFlow
  - PyTorch
- **Python Libraries:**
  - Librosa (audio processing)
  - Essentia (advanced audio analysis)
  - scikit-learn (classical ML methods)

---

## Progress

*Current status: The project is currently in the training and debugging phase. I have successfully implemented beat detection and initial spectral analysis. The deep learning model is under active development and debugging, with challenges related to data preprocessing and model convergence. Next steps include refining the training process, resolving debugging issues, and progressing to full integration tests for DAW output conversion.*

---

## Notes

- Start with short audio clips to manage data volume.
- Ensure precise beat detection to avoid misalignment in spectral analysis.
- Document each step for reproducibility and troubleshooting.
- Once a baseline model is achieved, explore additional musical attributes (harmony, dynamics, rhythm variations) for further analysis.

---

## Contact Information

For further discussion, collaboration, or a private demonstration of the system’s capabilities, please contact:
- **Email:** szaragoza2@wisc.edu  
- **LinkedIn:** ([https://www.linkedin.com/in/scott-zaragoza-198401329/](https://www.linkedin.com/in/scott-zaragoza-198401329/))
