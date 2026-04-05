## Refer to human_README.md for original writeup.

# Modern Hogwarts Sorting Classifier (DistilBERT)

A custom NLP text classification pipeline built with **DistilBERT** to categorize real-world, first-person personality descriptions into four modern psychological archetypes based loosely on the Harry Potter houses.

This project strips away the magical and fantasy elements entirely, focusing strictly on modern personality proxies (e.g., social dynamics, corporate behavior, life choices).

## 🔮 The 4 Archetypes (Classes)

*   **Label 0: Gryffindor (The Brave/Reckless):** Courageous, action-oriented, highly principled. Prone to acting before thinking, struggling with authority, and having a "hero complex."
*   **Label 1: Hufflepuff (The Loyal/Hardworking):** Dedicated, deeply fair, community-minded. Values team success over personal glory. Fatal flaws: being too trusting, avoiding conflict.
*   **Label 2: Ravenclaw (The Intellectual/Curious):** Analytical, witty, fiercely independent thinkers. Value knowledge/logic. Can come across as arrogant, aloof, or disconnected from practical emotions.
*   **Label 3: Slytherin (The Ambitious/Strategic):** Highly driven, resourceful, calculating. Networkers and hustlers. Fiercely loyal to a small inner circle, but can be ruthless or manipulative to outsiders.

## ⚙️ How It Works

The model pipeline is split into two main approaches:
1.  **Phase 1 (Zero-Shot Baseline):** Uses an out-of-the-box `typeform/distilbert-base-uncased-mnli` model to perform zero-shot classification based on the semantic archetype definitions.
2.  **Phase 2 (Fine-Tuning):** Trains a standard `distilbert-base-uncased` model via the Hugging Face `Trainer` API on a custom, labeled dataset to firmly map general language understanding to the 4 precise personality classes.

## 📊 The Dataset

*   **File:** `personality_classification_dataset.txt`
*   **Format:** Pipe-separated text (`text|label`), containing 3-5 sentence paragraphs written in the first person.
*   **Size:** 200 rows (50 paragraphs per class), ensuring a perfectly balanced dataset.

## 💻 Tech Stack
*   **Environment:** Python, Google Colab
*   **Libraries:** Hugging Face `transformers`, `datasets`, PyTorch, Scikit-learn, Pandas, NumPy

## 🚀 How to Run

1.  Open `diltilBERT_classifier.ipynb` in **Google Colab**.
2.  Upload `personality_classification_dataset.txt` directly into the main Google Colab file system (the `/content/` directory).
3.  Run all cells top-to-bottom.
4.  At the very end of the notebook, there is an interactive **Final Inference Tab** where you can type in your own distinct personality description and compare the guesses (and certainty percentages) from both the zero-shot and fine-tuned models side-by-side!

## 🤖 AI Transparency

Generative AI (specifically Claude) was transparently used in this project to assist with:
*   **Dataset Generation:** Creating 200 distinct, non-magical personality scenarios and mapping them properly to the class labels (manually writing these would be incredibly tedious).
*   **Redundant Tasks:** Scaffolding the PyTorch/Hugging Face dataset processing and standard boilerplate ML pipeline structures.