import gradio as gr
import insightface
from insightface.app import FaceAnalysis

wellcomingMessage = """
    <h1>Face Swapping</h1>
    <p>If you like this app, plase take a look at my <a href="https://www.meetup.com/tech-web3-enthusiasts-united-insightful-conversations/" target="_blank">Meetup Group</a>! There will be more interesting apps and events soon.</p>
    <p>Happy <span style="font-size:500%;color:red;">&hearts;</span> coding!</p>
    <div style="color: grey; font-size:small;">
        <p>🚀 Love my Face-Swapping Fun? Support Me with Crypto</p>
        <ul">
            <li>BTC: bc1q2m92e4hrtpk3keh2dsq8whljz7mfquv46xetwj</li>
            <li>ETH: 0x0459620D616C6D827603d43539519FA320B831c2</li>
        </ul>
    </div>
"""

assert insightface.__version__>='0.7'

value = 0
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

def swap_faces(faceSource, sourceFaceId, faceDestination, destFaceId):
    faces = app.get(faceSource)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    if len(faces) < sourceFaceId or sourceFaceId < 1:
        raise gr.Error(f"Source image only contains {len(faces)} faces, but you requested face {sourceFaceId}")
        
    source_face = faces[sourceFaceId-1]

    res_faces = app.get(faceDestination)
    res_faces = sorted(res_faces, key = lambda x : x.bbox[0])
    if len(res_faces) < destFaceId or destFaceId < 1:
        raise gr.Error(f"Destination image only contains {len(res_faces)} faces, but you requested face {destFaceId}")
    res_face = res_faces[destFaceId-1]

    result = swapper.get(faceDestination, res_face, source_face, paste_back=True)

    global value
    value = value + 1
    print(f"processed: {value}...")

    # for face in faces:
    #     res = swapper.get(res, face, source_face, paste_back=True)
    # cv2.imwrite("./t1_swapped.jpg", res)
    return result

gr.Interface(swap_faces, 
    [
        gr.Image(), 
        gr.Number(precision=0, value=1, info='face position (from left, starting at 1)'), 
        gr.Image(), 
        gr.Number(precision=0, value=1, info='face position (from left, starting at 1)')
    ],
     gr.Image(),
     description=wellcomingMessage,
     examples=[
            ['./Images/kim.jpg', 1, './Images/marilyn.jpg', 1],
            ['./Images/friends.jpg', 2, './Images/friends.jpg', 1],
        ],
).launch()
