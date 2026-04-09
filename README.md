# facerecserverless
This project implements a scalable, serverless face recognition pipeline using AWS services. It processes images asynchronously to detect and recognize faces using deep learning models.

This system decouples face detection and recognition into independent services using:
AWS Lambda (compute)
Amazon SQS (asynchronous communication)
Amazon S3 (model storage)

# Steps
- Client sends image (base64) to API Gateway
- Detection Lambda extracts faces using MTCNN
- Faces are sent to SQS queue
- Recognition Lambda processes each face
- Results are sent to response queue

# Flow
Client -> API Gateway -> Face Detection -> SQS Req Queue -> Face Recognition -> SQS Resp Queue
