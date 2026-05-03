import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
import os

app = FastAPI()

# Mount static files for the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Labels
LABEL_COLUMNS = ['humour', 'sarcasm', 'offensive', 'motivational']

# Setup device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

class MultimodalModel(nn.Module):
    def __init__(self, clip_model, num_labels):
        super().__init__()
        self.clip = clip_model
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_features = self.clip.visual_projection(vision_outputs.pooler_output)
        
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = self.clip.text_projection(text_outputs.pooler_output)
        
        fused = torch.cat((image_features, text_features), dim=1)
        logits = self.fc(fused)
        return logits

model = MultimodalModel(clip_model, len(LABEL_COLUMNS)).to(device)

# Load the weights created in colab
MODEL_PATH = "multimodal_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model weights loaded successfully.")
else:
    print(f"Warning: {MODEL_PATH} not found. Please place your .pth file in the same directory as this script.")

model.eval()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(text: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(text=[text], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values']
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        results = {LABEL_COLUMNS[i]: float(probs[i]) for i in range(len(LABEL_COLUMNS))}
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
