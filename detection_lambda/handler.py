import os
from io import BytesIO
import json
import boto3
import base64
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# Initialize MTCNN
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

# SQS Setup
sqsClient = boto3.client("sqs", region_name="us-east-1")
requestQueueUrl = os.environ.get("REQUEST_QUEUE_URL")

# Send detected faces for recognition
def sendToSQS(detectedFaces, request_id, filename):

    for face in detectedFaces:
        message = {
            'request_id': request_id,
            'filename': filename,
            'content': face
        }
        
        sqsClient.send_message(
            QueueUrl=requestQueueUrl,
            MessageBody=json.dumps(message)
        )

def faceDetection(filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.array(img)
    img = Image.fromarray(img)

    key = os.path.splitext(os.path.basename(filePath))[0].split(".")[0]

    # Detect faces using MTCNN
    faces, prob = mtcnn(img, return_prob=True, save_path=None)

    if faces is not None:
        if isinstance(faces, torch.Tensor) and len(faces.shape) == 3:
            faces = faces.unsqueeze(0)

        isolatedFaces = []

        for i, face in enumerate(faces):
            faceImage = face.clone()
            for c in range(3):
                channel = faceImage[c]
                faceImage[c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            faceImage = (faceImage * 255).byte()

            faceImage = faceImage.permute(1, 2, 0).numpy()
            facePIL = Image.fromarray(faceImage, mode="RGB")

            # encoding for base64 before transfer
            buffered = BytesIO()
            facePIL.save(buffered, format="JPEG")
            encodedFace = base64.b64encode(buffered.getvalue()).decode("utf-8")
            isolatedFaces.append(encodedFace)

        return isolatedFaces
    else:
        print("No faces detected.")
        return None

def handler(event, context):
    try:
        body = json.loads(event['body'])
        imageData = body['content']
        request_id = body['request_id']
        filename = body['filename'].split(":")[0]

        # Decode the base64 image data
        decodedImage = base64.b64decode(imageData)
        filePath = f"/tmp/{filename}"
        with open(filePath, 'wb') as f:
            f.write(decodedImage)

        # Face detection
        detectedFaces = faceDetection(filePath)

        if detectedFaces is not None:
            # Send detected faces to the request queue
            sendToSQS(detectedFaces, request_id, filename)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Face detected and sent to recognition queue.',
                    'request_id': request_id
                })
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'No faces detected.'})
            }
    except Exception as e:
        print(f"Error: {e}")
        return None
