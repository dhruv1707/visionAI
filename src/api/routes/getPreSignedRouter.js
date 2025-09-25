// import multer from "multer"
// import multerS3 from "multer-s3"
// import { s3 } from "../index.js"
import express, { Router } from "express"
import * as getPreSignedController from "../controllers/getPreSignedController.js"

const router = Router()

router.route("/")
    .post(getPreSignedController.getPreSignedURL)

// const upload = multer({
//     storage: multerS3({
//         s3: s3,
//         bucket: process.env.AWS_BUCKET,
//         metadata: function(req, file, cb) {
//             cb(null, {fieldName: file.fieldname});
//         },
//         key: function(req, file, cb) {
//             cb(null, Date.now().toString())
//         }

//     })
// })
// const storage = multer.diskStorage({
//     destination: function(req, file, cb){
//         cb(null, "./uploads")
//     },
//     filename: function(req, file, cb) {
//         cb(null, file.originalname)
//     },
// })
// const upload = multer({ storage:storage })

// router.route("/")
//     .post(upload.array('folder'), uploadController)
//     // form input field name should be folder

export default router

