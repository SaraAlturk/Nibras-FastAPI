from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import io

app = FastAPI()

# Allow React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (you can restrict to your frontend URL later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor once at startup
saved_dir = "./model"  # your model folder
processor = Wav2Vec2Processor.from_pretrained(saved_dir)
model = Wav2Vec2ForSequenceClassification.from_pretrained(saved_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict(audio_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    target_sr = processor.feature_extractor.sampling_rate
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[pred_id]

# API endpoint
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    prediction = predict(audio_bytes)
    return {"prediction": prediction}
