
# app/predict_api.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchaudio
import io
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

router = APIRouter()

# ————— Configuration ——————————————————————————————————————————

# Path to your fine-tuned model directory
MODEL_PATH = "/Users/ayaaldoubi/Desktop/nibras_api/model"

# The sampling rate your model expects
TARGET_SR = 16_000

# Device selection (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ID→label mapping (should match your model config)
ID2LABEL = {
    0: "S",
    1: "W",
    2: "PH",
    3: "PR",
    4: "none"
}

# ————— Load processor + model ——————————————————————————————————

logging.basicConfig(level=logging.INFO)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ————— Utility to load & preprocess audio —————————————————————————————————

def load_audio_from_bytes(file_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(file_bytes))
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample to TARGET_SR
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    return waveform.squeeze().cpu().numpy()

# ————— Prediction endpoint ——————————————————————————————————————————

@router.post("/", summary="Predict stutter type from uploaded audio")
async def predict_stutter_type(audio_file: UploadFile = File(...)):
    # Validate MIME type
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    
    data = await audio_file.read()
    try:
        audio = load_audio_from_bytes(data)
    except Exception as e:
        logging.error(f"Audio loading failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")
    
    # Tokenize → model → softmax
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).cpu().item())
        pred_label = ID2LABEL.get(pred_idx, "Unknown")
    
    return JSONResponse({
        "prediction": pred_label,
        "prediction_index": pred_idx,
        "confidence": probs[pred_idx],
        "all_confidences": {ID2LABEL[i]: probs[i] for i in range(len(probs))}
    })
