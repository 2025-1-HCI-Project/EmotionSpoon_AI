import pandas as pd
from dotenv import load_dotenv
import pymysql
import os

class MusicDatabase:
    def __init__(self):
        load_dotenv()

        conn = pymysql.connect(
            host=os.getenv('HCI_HOST'),
            user=os.getenv('HCI_USER'),
            password=os.getenv('HCI_PASSWORD'),
            port=3306
        )

        self.cursor = conn.cursor()

        self.cursor.execute("USE " + os.getenv('HCI_USER'))
        print("Music database connected")
    
    def Retrieve(self):
        query = "SELECT * FROM music"
        
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        
        return pd.DataFrame(data, columns=["id", "artist", "song", "lyric", "emotion", "lyric_embedding", "emotion_embedding", "link"])
