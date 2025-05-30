import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import torch, os

from EmotionSpoon_AI import music_database

class SongRecommender:
    def __init__(self):
        load_dotenv()

        # .env 파일에 있는 hugging face token 로드
        token = os.getenv('HUGGING_FACE_HUB_TOKEN')

        GPU_NUM = 0
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        print('Device for Embedding:', device)

        if torch.cuda.is_available():
            torch.cuda.set_device(device)

            print('Current cuda device:', torch.cuda.current_device())
            print('Count of using GPUs:', torch.cuda.device_count())

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})

        self.music_database = music_database.MusicDatabase().Retrieve()
        print("Embedding model loaded")

    # 완전 동일하면 1, orthogonal하면 0, 반대 의미의 경우 -1
    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def recommend(self, context: str, emotion: str) -> dict:
        music_playlist = []

        emotion_embedding = self.embedding_model.embed_query(emotion)
        context_embedding = self.embedding_model.embed_query(context)
        for i in range(len(self.music_database)):
            music_data = self.music_database.loc[i].to_dict()

            emotion_dist = self._cosine_similarity(emotion_embedding, eval(music_data["emotion_embedding"]))
            context_dist = self._cosine_similarity(context_embedding, eval(music_data["lyric_embedding"]))

            # 감정 : 맥락 = 3 : 7 ensemble
            music_playlist.append({
                "artist": music_data["artist"],
                "song": music_data["song"],
                "lyric": music_data["lyric"],
                "link": music_data["link"],
                "similarity": 3 * emotion_dist + 7 * context_dist
            })
        
        music_playlist.sort(key=lambda music: music["similarity"], reverse=True)
        
        return music_playlist[:10]