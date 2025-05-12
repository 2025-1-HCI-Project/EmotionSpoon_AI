from paddleocr import PaddleOCR
from PIL import Image
import cv2  # OpenCV 사용 (이미지 로드 및 자르기)

class TextRecognizer:
    def __init__(self):
    # need to run only once to download and load model into memory
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    # Paddleocr supports Chinese, English, French, German, Korean and Japanese
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
    # to switch the language model in order
    def image_to_text(self, img_path: str) -> str:
        original_image = cv2.imread(img_path)
        result = self.ocr.ocr(original_image, cls=True)

        # draw result
        txts = [line[1][0] for line in result[0]]
        
        return "\n".join(txts)