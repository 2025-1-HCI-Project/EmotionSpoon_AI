import EmotionSpoon_AI.sentiment_analyzer
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn
import uuid, io, os

from EmotionSpoon_AI import text_recognizer, sentiment_analyzer, music_recommender

OCR = text_recognizer.TextRecognizer()
analyzer = sentiment_analyzer.SentimentAnalyzer()
recommender = music_recommender.SongRecommender()

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

class PostRequest(BaseModel):
    input_data: str

@app.get("/test")
async def test():
    return {"test": "connection!"}

@app.post("/recommend")
async def analyze_sentiment(request: PostRequest):
    sentiment = analyzer.analyze(request.input_data)
    song = recommender.recommend(context=request.input_data, emotion=sentiment)
    return {"sentiment": sentiment, **song}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ocr")
async def upload_image(file: UploadFile = File(...)):
    # 파일 유효성 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다")
    
    # 파일 저장
    file_ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # 이미지 처리
    try:
        contents = await file.read()
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()  # 이미지 손상 확인
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        return {"text": OCR.image_to_text(file_path)}
    except Exception as e:
        raise HTTPException(500, f"오류: {str(e)}")

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=5050)