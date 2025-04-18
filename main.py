''' -------------------- < LLM > -------------------- '''
import torch, os
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일 로드
load_dotenv()

# .env 파일에 있는 hugging face token 로드
token = os.getenv('HUGGING_FACE_HUB_TOKEN')

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

if torch.cuda.is_available():
  torch.cuda.set_device(device)

  print('Current cuda device:', torch.cuda.current_device())
  print('Count of using GPUs:', torch.cuda.device_count())

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config).to(device)
else:
  model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=1e-6,
    return_full_text=False,
    max_new_tokens=64
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

PROMPT = \
'''
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a psychotherapist to analyze sentiment in Text.

Guideline:
    - Extract one of main emotion from Text.
    - The emotion cannot be general sentiment such as good, bad, positive, and negative.

Remember, the emotion must be detailed as you can.

Print the main emotion without your rationale.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Text: {input}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Sentiment:
'''.strip()

prompt_template = PromptTemplate(
    input_variables=["input"],
    template=PROMPT,
)

llm_chain = prompt_template | llm | StrOutputParser()

print("LLM chain loaded")

''' -------------------- < Embedding > -------------------- '''

import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

# 완전 동일하면 1, orthogonal하면 0, 반대 의미의 경우 -1
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})

lyric_dataset = pd.read_csv("lyric_dataset.csv")

def RecommendSong(context: str, emotion: str) -> dict:
  max_index = 0
  max_dist = float('-inf')

  emotion_embedding = embedding_model.embed_query(emotion)
  context_embedding = embedding_model.embed_query(context)
  for i in range(len(lyric_dataset)):
    emotion_dist = cosine_similarity(emotion_embedding, eval(lyric_dataset.loc[i, "emotion_embedding"]))
    context_dist = cosine_similarity(context_embedding, eval(lyric_dataset.loc[i, "lyric_embedding"]))

    # 감정 : 맥락 = 3 : 7 ensemble
    if max_dist < 3 * emotion_dist + 7 * context_dist:
       max_dist = 3 * emotion_dist + 7 * context_dist
       max_index = i
  
  return lyric_dataset.loc[max_index].to_dict()

print("Embedding model loaded")

''' -------------------- < OCR > -------------------- '''

from paddleocr import PaddleOCR
from PIL import Image
import cv2  # OpenCV 사용 (이미지 로드 및 자르기)

# need to run only once to download and load model into memory
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Paddleocr supports Chinese, English, French, German, Korean and Japanese
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order
def _image_to_text(img_path: str) -> str:
  original_image = cv2.imread(img_path)
  result = ocr.ocr(original_image, cls=True)

  # draw result
  txts = [line[1][0] for line in result[0]]
  
  return "\n".join(txts)

''' -------------------- < API > -------------------- '''

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid, io

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

@app.post("/run")
async def analyze_sentiment(request: PostRequest):
    sentiment = llm_chain.invoke({"input": request.input_data}).split("\n")[0].strip()
    song = RecommendSong(request.input_data, sentiment)
    return {"sentiment": sentiment, **song}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
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
        return {"text": _image_to_text(file_path)}
    except Exception as e:
        raise HTTPException(500, f"오류: {str(e)}")

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=5050)