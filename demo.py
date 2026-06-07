import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import argparse

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def build_model(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()
print(f"Model loaded from {args.checkpoint} | Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_genre(audio_path):
    duration_total = librosa.get_duration(path=audio_path)
    offset = max(0, duration_total / 2 - 1.5)
    audio, sr = librosa.load(audio_path, sr=22050, offset=offset, duration=3.0)
    target_len = 22050 * 3
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time',
                                    y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title("Mel Spectrogram")
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=96)
    plt.close(fig)
    buf.seek(0)
    spec_img = Image.open(buf).copy()
    buf.close()
    mel_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    pil_img = Image.fromarray((mel_normalized * 255).astype(np.uint8)).convert('RGB')
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    confidences = {GENRES[i]: float(probs[i]) for i in range(len(GENRES))}
    top_genre = GENRES[probs.argmax()]
    top_conf = probs.max()
    result = f"Predicted genre: {top_genre.upper()} ({top_conf*100:.1f}% confidence)"
    return result, confidences, spec_img

with gr.Blocks(title="TensorTunes") as demo:
    gr.Markdown("# TensorTunes - Music Genre Classifier")
    gr.Markdown("Upload a .wav or .mp3 file and ResNet18 will predict its genre.")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload audio file")
    predict_btn = gr.Button("Predict Genre", variant="primary")
    with gr.Row():
        result_text = gr.Markdown()
        spec_output = gr.Image(label="Mel Spectrogram")
    confidence_output = gr.Label(num_top_classes=5, label="Top 5 Predictions")
    predict_btn.click(
        fn=predict_genre,
        inputs=[audio_input],
        outputs=[result_text, confidence_output, spec_output]
    )
    gr.Markdown("Model: ResNet18 fine-tuned on GTZAN | Accuracy: 66.9% test")

demo.launch()