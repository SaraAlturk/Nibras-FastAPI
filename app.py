from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import io

app = FastAPI()

# Allow React frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor
saved_dir = "./saved_model"
processor = Wav2Vec2Processor.from_pretrained(saved_dir)
model = Wav2Vec2ForSequenceClassification.from_pretrained(saved_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # Load audio from bytes
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Resample if needed
        target_sr = processor.feature_extractor.sampling_rate
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

        # Make mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Prepare input
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits

        pred_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[pred_id]

        return {"prediction": label}

    except Exception as e:
        return {"error": str(e)}
