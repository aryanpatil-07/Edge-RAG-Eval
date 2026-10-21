require("dotenv").config();

const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const noteRoutes = require("./routes/noteRoutes");
const aiRoute = require("./routes/aiRoute");
const chatRoutes = require("./routes/chatRoutes");

const app = express();
const uploadsDir = path.join(__dirname, "uploads");

if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

app.use(cors());
app.use(express.json());
app.use("/uploads", express.static(uploadsDir));

// Routes
app.use("/notes", noteRoutes);
app.use("/api/ai", aiRoute);
app.use("/api/chat", chatRoutes);

// DB
const mongoUri = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/secondbrain";
if (mongoUri && !mongoUri.includes("<username>")) {
  mongoose.connect(mongoUri)
    .then(() => console.log("Connected Successfully to MongoDB"))
    .catch(err => console.warn("MongoDB connection warning:", err.message));
} else {
  console.log("No valid MONGODB_URI configured in .env. Waiting for database credentials.");
}

// Root
app.get("/", (req, res) => {
  res.send("API Running");
});

// Server
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on PORT ${PORT}`);
});