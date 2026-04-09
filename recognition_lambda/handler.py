import boto3
import base64
import torch
import json
import os
import numpy as np
from io import BytesIO
from facenet_pytorch import InceptionResnetV1
from PIL import Image

resnet = InceptionResnetV1(pretrained='vggface2').eval()

# S3 Setup
s3Client = boto3.client('s3', region_name="us-east-1")
s3BucketName = os.environ.get("BUCKET_NAME")
s3Key = 'data.pt'

# Download model weights from S3
def downloadWeights(s3BucketName, s3Key, local_file_path='/tmp/data.pt'):
    try:
        # Download to Lambda /tmp directory
        s3Client.download_file(s3BucketName, s3Key, local_file_path)
        print(f"Downloaded {s3Key} to {local_file_path}")
    except Exception as e:
        print(f"Couldn't download weights: {e}")
        raise

# SQS Setup
sqsClient = boto3.client("sqs", region_name="us-east-1")
requestQueueUrl = "https://sqs.us-east-1.amazonaws.com/248189923120/1233320792-req-queue"
responseQueueUrl = "https://sqs.us-east-1.amazonaws.com/248189923120/1233320792-resp-queue"

# Send recognition results to response queue
def sendToSQS(response):
    sqsClient.send_message(
        QueueUrl=responseQueueUrl,
        MessageBody=json.dumps(response)
    )

def faceRecognition(filePath, resnetModel):

    facePIL = Image.open(filePath).convert("RGB")
    key = os.path.splitext(os.path.basename(filePath))[0].split(".")[0]
    numpyFace = np.array(facePIL, dtype=np.float32)
    numpyFace /= 255.0

    numpyFace = np.transpose(numpyFace, (2, 0, 1))
    faceTensor = torch.tensor(numpyFace, dtype=torch.float32)

    saved_data = torch.load('/tmp/data.pt')

    if faceTensor != None:
        emb             = resnetModel(faceTensor.unsqueeze(0)).detach()
        embedding_list  = saved_data[0]  
        name_list       = saved_data[1]  
        dist_list       = [] 

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))
        print(idx_min)
        return name_list[idx_min]
    else:
        return ["Unknown Person! Could not recognize!"]

def handler(event, context):
    try:
        recognitionRequest = event['Records'][0]['body']
        requestData = json.loads(recognitionRequest)
        
        request_id = requestData['request_id']
        filename = requestData['filename'].split(":")[0]
        ImageData = requestData['content']

        downloadWeights(s3BucketName, s3Key)

        # Decode the base64 image data
        decoded_image_data = base64.b64decode(ImageData)

        filePath = f"/tmp/{filename}"
        with open(filePath, 'wb') as f:
            f.write(decoded_image_data)

        # Face recognition using the S3 model weights
        results = faceRecognition(filePath, resnet)

        response = {
            'request_id': request_id,
            'result': results
        }

        sendToSQS(response)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Face recognized! Results stored!'})
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return None
