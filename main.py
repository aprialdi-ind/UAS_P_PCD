import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io


app = FastAPI()

# Izinkan CORS agar Frontend bisa mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    method: str = Form(...),
    threshold: int = Form(...)
):
    # Baca gambar
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Pre-processing: Blur untuk mengurangi noise
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    
    result = None

    if method == "Sobel":
        sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(sobelx, sobely)

    elif method == "Laplace":
        result = cv2.Laplacian(img_blur, cv2.CV_64F)
        result = np.absolute(result)

    elif method == "Robert":
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_x = cv2.filter2D(img_blur, -1, kernel_x)
        grad_y = cv2.filter2D(img_blur, -1, kernel_y)
        result = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)

    elif method == "Prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        grad_x = cv2.filter2D(img_blur, -1, kernel_x)
        grad_y = cv2.filter2D(img_blur, -1, kernel_y)
        result = cv2.magnitude(grad_x.astype(float), grad_y.astype(float))

    elif method == "Frei-Chen":
        sqrt2 = np.sqrt(2)
        # Kernel Frei-Chen untuk deteksi tepi (orthogonal)
        kernel_x = np.array([[1, sqrt2, 1], [0, 0, 0], [-1, -sqrt2, -1]], dtype=np.float32) / (2 + sqrt2)
        kernel_y = np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]], dtype=np.float32) / (2 + sqrt2)
        grad_x = cv2.filter2D(img_blur, -1, kernel_x)
        grad_y = cv2.filter2D(img_blur, -1, kernel_y)
        result = cv2.magnitude(grad_x.astype(float), grad_y.astype(float))

    # Normalisasi hasil ke 0-255
    if result is not None:
        result = np.uint8(np.absolute(result))
        # Terapkan Thresholding
        _, final_result = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
    else:
        final_result = img

    base64_str = image_to_base64(final_result)
    return {"result": base64_str}

if __name__ == "__main__":
    # Render akan memberikan port lewat environment variable PORT
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)