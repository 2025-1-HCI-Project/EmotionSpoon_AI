# EmotionSpoon_AI

## Project Overview

EmotionSpoon_AI is an intelligent system utilizing emotion recognition technology in the field of Human-Computer Interaction (HCI). The project aims to optimize interactions by recognizing users’ emotional states through various biosignals and expressions.

## Key Features

- **Multimodal Emotion Recognition**: Integrates multiple input channels such as hand-written text for emotion detection.
- **Real-time Emotion Analysis**: Detects and analyzes users’ emotional states in real time.
- **Emotion-based Interaction**: Generates context-appropriate playlist based on the recognized emotions.
- **User-independent Models**: Provides consistent emotion recognition performance regardless of individual differences.

## Tech Stack

- **Language**: Python 3.11
- **Large Language Model**: Llama-3.2-1B
- **Sentence Embedding Model**: all-MiniLM-L6-v2
- **Optical Character Recognition**: PaddleOCR
- **Data Processing**: NumPy, Pandas

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/2025-1-HCI-Project/EmotionSpoon_AI.git
    cd EmotionSpoon_AI
    ```

2. Install required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

Basic execution:
    ```
    python main.py
    ```

## Emotion Recognition Methodology

This project uses the following emotion models:

- **Persona Prompt**: Classifies basic emotion categories such as happiness, sadness, surprise, fear, anger, and disgust as if model were psychotherapist.
- **Zero-Shot Prompting**: Categorizes emotions with high-flexibility and cost-efficiency.

## Future Development Plans

- Improve emotion recognition models considering diverse backgrounds
- Optimize real-time processing performance
- Implement emotion recognition API endpoints

## Contribution Guidelines

1. Create a new issue or review existing issues
2. Fork the repository and create a development branch
3. Implement and test your changes
4. Submit a Pull Request

## Contact

For project-related inquiries, please submit an issue via GitHub.

---

This project was developed as part of the 2025 Human-Computer Interaction (HCI) course. The goal is to realize more empathetic and intelligent interfaces through emotion recognition technology.