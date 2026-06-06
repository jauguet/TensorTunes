# TensorTunes 🎵
### Music Genre Classification with Deep Learning
**DL2026 — Beijing Jiaotong University**

## Results
| Model | Test Accuracy |
|---|---|
| SVM Baseline (MFCC + features) | 74.0% |
| ResNet18 Fine-tuned (Mel Spectrograms) | 66.9% |

## Reproduce in 5 steps
1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/TensorTunes
cd TensorTunes
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download GTZAN dataset from Kaggle
```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d ./gtzan
```
4. Generate mel spectrograms
```bash
python preprocess.py
```
5. Evaluate the model
```bash
python evaluate.py
```

## Hardware
- Trained on Google Colab (T4 GPU)
- Training time : ~15 minutes
- Seed : 42

## License
MIT
