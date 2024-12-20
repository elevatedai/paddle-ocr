from fastapi import FastAPI, Request, HTTPException
from paddleocr import PaddleOCR, __version__ as paddleocr_version
import io
from PIL import Image
import numpy as np
from pydantic import BaseModel, HttpUrl
import logging
from urllib.parse import urlparse
import aiohttp
import logging
import os
import re

logging.getLogger('ppocr').setLevel(logging.ERROR)

app = FastAPI(title="OCR API", description="API for extracting text from images using PaddleOCR")

ocr = PaddleOCR(use_angle_cls=True, lang='en')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageInput(BaseModel):
    url: HttpUrl

async def download_image(url: str) -> bytes:
    """Download image from URL asynchronously"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch image from URL")
                return await response.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

def process_image(image_data: bytes) -> list:
    """Process image data and return OCR results"""
    image = Image.open(io.BytesIO(image_data))
    img_array = np.array(image)
    try:
       result = ocr.ocr(img_array,cls=True)     
       finaltext = ""
       for line in result[0]:
           text = line[1][0]
           finaltext+=text    
       return re.sub(r"[^a-zA-Z0-9]", "", finaltext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/extract_text/file")
async def extract_text_from_file(request: Request):
    """Extract text from uploaded image file"""
    try:
        contents = await request.body()

        extracted_text = process_image(contents)
        
        return {
            "status": "success",
            "source": "file",
            "data": extracted_text
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_text/url")
async def extract_text_from_url(image_input: ImageInput):
    """Extract text from image URL"""
    try:
        url = str(image_input.url)
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL")
        
        image_data = await download_image(url)

        extracted_text = process_image(image_data)
        
        return {
            "status": "success",
            "source": "url",
            "data": extracted_text
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_engine": "PaddleOCR",
        "version":  paddleocr_version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get("PORT", 8000))