import { s3 } from "../index.js"
import { CreateMultipartUploadCommand, UploadPartCommand, CompleteMultipartUploadCommand, PutObjectCommand } from "@aws-sdk/client-s3"
import { getSignedUrl } from "@aws-sdk/s3-request-presigner"

export const getPreSignedURL = async(req, res) => {
    const {userId, files} = req.body
    if (!userId || !Array.isArray(files)) {
        return res.status(400).json({ error: 'Missing userId or files array' });
    }
    const THRESHOLD_SIZE = 100*1024*1024
    const uploads = []

    for (const file of files){
        if (file.size > THRESHOLD_SIZE){
            try {
                let partSize = 20*1024*1024 // 20MiB
                
                const {relativePath, contentType, size} = file
                if (!relativePath || !contentType || typeof size !== 'number') {
                    return res.status(400).json({ error: 'Missing required fields' });
                }
                const key = `users/${userId}/${relativePath.replace(/^\/+/, '')}`
                if (partSize < 5*1024*1024){
                    partSize = 5*1024*1024
                }
                const numParts = Math.ceil(file.size / partSize)
                const createUploadCommand = new CreateMultipartUploadCommand({
                    Bucket: process.env.AWS_BUCKET,
                    Key: key
                })
                const {uploadId} = await s3.send(createUploadCommand)
                const parts = []
                for (let i=1; i <= numParts; i++) {
                    const command = new UploadPartCommand({
                        Bucket: process.env.AWS_BUCKET,
                        Key: key,
                        UploadId: uploadId,
                        PartNumber: i
                    })
                    const url = await getSignedUrl(s3, command, {expiresIn: 900})
                    parts.push({
                        "partNumber": i,
                        "url": url
                    })
                }
                uploads.push({
                    "key": key,
                    "type": "multipart",
                    "uploadId": uploadId,
                    "partSize": partSize,
                    "contentType": contentType,
                    "parts": parts
                })
                
            } catch (error) {
                console.error("Error generating a presigned url: ", error)
                return {
                    relativePath,
                    error: error.message || 'Failed to generate presigned data',
                }
            }
        }
        else {
            // Single part upload
            const {relativePath, contentType, size} = file
            if (!relativePath || !contentType || typeof size !== 'number') {
                return res.status(400).json({ error: 'Missing required fields' });
            }
            const key = `users/${userId}/${relativePath.replace(/^\/+/, '')}`
            const putObject = new PutObjectCommand({
                Bucket: process.env.AWS_BUCKET,
                Key: key,
                ContentType: contentType
            })
            const url = await getSignedUrl(s3, putObject, {expiresIn: 900})
            uploads.push({
                "key": key,
                "type": "single",
                "url": url,
                "contentType": contentType
            })
        }
    }
    return res.status(200).json({uploads})


}