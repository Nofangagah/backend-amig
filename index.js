const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const { v4: uuidv4 } = require("uuid");
const { Firestore } = require("@google-cloud/firestore");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(cors());
const PORT = process.env.PORT || 8080;
const MAX_FILE_SIZE = 1000000; // 1MB
const modelUrl = "https://storage.googleapis.com/submission-mlgc-nofan/submissions-model%20(1)/model.json";

const firestore = new Firestore({
    projectId: process.env.PROJECT_ID,
    keyFilename: process.env.GOOGLE_CLOUD_CREDENTIALS,
    databaseId: process.env.DATABASE_ID
});

console.log("Using project ID:", process.env.PROJECT_ID);
console.log("Using credentials:", process.env.GOOGLE_CLOUD_CREDENTIALS);

let model;

// Memuat model saat aplikasi dimulai
(async () => {
    try {
        console.log("Memuat model...");
        model = await tf.loadGraphModel(modelUrl);
        console.log("Model berhasil dimuat.");
    } catch (error) {
        console.error("Gagal memuat model:", error.message);
    }
})();

// Konfigurasi Multer untuk unggahan gambar
const upload = multer({
    limits: { fileSize: MAX_FILE_SIZE },
    fileFilter(req, file, cb) {
        if (!file.mimetype.startsWith("image/")) {
            return cb(new Error("File harus berupa gambar."), false);
        }
        cb(null, true);
    },
});

// Endpoint untuk prediksi gambar
app.post("/predict", upload.single("image"), async (req, res) => {
    if (!model) {
        return res.status(500).json({
            status: "fail",
            message: "Model belum siap untuk digunakan.",
        });
    }

    try {
        if (!req.file || !req.file.buffer) {
            return res.status(400).json({
                status: "fail",
                message: "Tidak ada file yang diunggah atau file kosong.",
            });
        }

        // Proses gambar menjadi tensor
        const imageBuffer = req.file.buffer;
        const tensor = tf.node
            .decodeImage(imageBuffer)
            .resizeBilinear([224, 224])
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255.0));

        // Prediksi menggunakan model
        const prediction = await model.predict(tensor).dataSync();
        const threshold = 0.58; // Ambang batas untuk deteksi kanker
        const probability = prediction[0];
        const result = probability > threshold ? "Cancer" : "Non-cancer";

        // Struktur respons sesuai spesifikasi
        const response = {
            status: "success",
            message: "Model is predicted successfully",
            data: {
                id: uuidv4(),
                result: result,
                suggestion: result === "Cancer" ? "Segera periksa ke dokter!" : "Penyakit kanker tidak terdeteksi.",
                createdAt: new Date().toISOString(),
            },
        };

        // Simpan hasil prediksi ke Firestore
        await firestore.collection("predictions").add(response.data);

        // Kirim respons
        res.status(201).json(response);
    } catch (error) {
        console.error("Error during prediction:", error.message);
        res.status(400).json({
            status: "fail",
            message: "Terjadi kesalahan dalam melakukan prediksi",
        });
    }
});

// Endpoint untuk mendapatkan riwayat prediksi
app.get("/predict/histories", async (req, res) => {
    try {
        const snapshot = await firestore.collection("predictions").get();
        const histories = snapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data(),
        }));
        res.json({
            status: "success",
            data: histories,
        });
    } catch (error) {
        console.error("Error fetching histories:", error.message);
        res.status(500).json({
            status: "error",
            message: "Terjadi kesalahan saat mengambil data riwayat prediksi.",
        });
    }
});

// Middleware untuk menangani error dari Multer
app.use((err, req, res, next) => {
    if (err.code === "LIMIT_FILE_SIZE") {
        return res.status(413).json({
            status: "fail",
            message: "Payload content length greater than maximum allowed: 1000000",
        });
    }
    if (err.message === "File harus berupa gambar.") {
        return res.status(400).json({
            status: "fail",
            message: "File harus berupa gambar.",
        });
    }
    console.error("Unhandled error:", err.message);
    res.status(500).json({
        status: "error",
        message: "Terjadi kesalahan pada server.",
    });
});

// Menjalankan server
app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
});
