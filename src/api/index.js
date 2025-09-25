import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import { S3Client } from "@aws-sdk/client-s3";
import getPreSignedRouter from "./routes/getPreSignedRouter.js"

export const s3 = new S3Client({
    credentials:{
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    },
    region: process.env.AWS_REGION
})


const PORT = process.env.PORT || 8000;
const app = express()

dotenv.config()
app.use(cors())
app.use(express.json())

app.use("/api/get_presigned", getPreSignedRouter)

app.listen(PORT, () => {
    console.log(`Server listening on PORT: ${PORT}`)
})
