import gradio as gr
import insightface
from insightface.app import FaceAnalysis
import os

# Reduce model size if possible
assert insightface.__version__ >= '0.7'

# Cache directory for models if required
MODEL_CACHE_DIR = "./models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Welcome message
wellcomingMessage = """
    <h1>Face Swapping</h1>
    <p>If you like this app, please take a look at my <a href="https://www.meetup.com/tech-web3-enthusiasts-united-insightful-conversations/" target="_blank">Meetup Group</a>!</p>
    <p>Happy <span style="font-size:500%;color:red;">&hearts;</span> coding!</p>
"""

# Set up Face Analysis and the face swapping model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load model from cache or from the source
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True, download_dir=MODEL_CACHE_DIR)

# Global count for processed images
value = 0

# Function to perform face swap
def swap_faces(faceSource, sourceFaceId, faceDestination, destFaceId):
    faces = app.get(faceSource)
    faces = sorted(faces, key=lambda x: x.bbox[0])  # Sort faces based on left position
    if len(faces) < sourceFaceId or sourceFaceId < 1:
        raise gr.Error(f"Source image only contains {len(faces)} faces, but you requested face {sourceFaceId}")
    
    source_face = faces[sourceFaceId - 1]

    res_faces = app.get(faceDestination)
    res_faces = sorted(res_faces, key=lambda x: x.bbox[0])
    if len(res_faces) < destFaceId or destFaceId < 1:
        raise gr.Error(f"Destination image only contains {len(res_faces)} faces, but you requested face {destFaceId}")
    
    res_face = res_faces[destFaceId - 1]
    
    result = swapper.get(faceDestination, res_face, source_face, paste_back=True)

    global value
    value += 1
    print(f"Processed: {value} images...")

    return result

# Gradio Interface
gr.Interface(
    swap_faces, 
    [
        gr.Image(type="pil", image_mode="RGB"), 
        gr.Number(precision=0, value=1, info='Face position (from left, starting at 1)'), 
        gr.Image(type="pil", image_mode="RGB"), 
        gr.Number(precision=0, value=1, info='Face position (from left, starting at 1)')
    ], 
    gr.Image(),
    description=wellcomingMessage,
    examples=[
        ['./Images/kim.jpg', 1, './Images/marilyn.jpg', 1],
        ['./Images/friends.jpg', 2, './Images/friends.jpg', 1],
    ],
    queue=True,  # Limit concurrent users
    server_port=7860  # Optional: Set a specific port for Railway
).launch()
