import pdfplumber
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import io

def extract_text(file_bytes: bytes, content_type: str) -> str:
    if "pdf" in content_type:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    elif "image" in content_type:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)
    elif "text" in content_type:
        return file_bytes.decode("utf-8", errors="ignore")
    return ""