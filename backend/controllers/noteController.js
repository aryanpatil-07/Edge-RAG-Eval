/**
 * @file noteController.js
 * @concept CRUD operations (Mongo)
 * @concept RESTful endpoint design
 * @concept HTTP status codes used correctly (201 Created, 404 Not Found, 500 Internal Error)
 * @concept Server-side error handling
 * @concept JavaScript — Event loop (setImmediate background execution)
 * @concept JavaScript — async/await
 */

const mongoose = require("mongoose");
const Note = require("../models/Note");
const Chat = require("../models/Chat");
const fs = require("fs");
const path = require("path");
const { extractTextFromImage } = require("../services/ocrService");
const { analyzeNote } = require("../services/analysisService");

const isDbConnected = () => mongoose.connection.readyState === 1;

// Resilient in-memory storage used when MongoDB is offline / credentials pending
let memoryNotes = [
  {
    _id: "demo-note-1",
    title: "React Component Architecture & Hooks",
    content: "## Core Principles\n\nReact components are pure functions of props and state.\n\n### Essential Hooks\n- **useState**: local state management\n- **useEffect**: side effects and lifecycle synchronization\n- **useMemo / useCallback**: memoization and performance optimization\n- **useRef**: mutable references across renders without triggering repaints\n\n```jsx\nfunction Counter() {\n  const [count, setCount] = useState(0);\n  return <button onClick={() => setCount(c => c + 1)}>Count: {count}</button>;\n}\n```",
    tags: ["react", "javascript", "frontend"],
    sourceType: "text",
    processingStatus: "processed",
    createdAt: new Date().toISOString(),
    analysis: {
      score: 84,
      level: "intermediate",
      summary: "Strong conceptual understanding of component design, React hooks, and render performance.",
      strengths: ["Clean understanding of hook rules", "Solid state lifecycle reasoning", "Practical code structure"],
      gaps: ["Concurrent rendering & Suspense", "Custom hook abstractions", "SSR hydration patterns"],
      currentNode: "n2",
      nodes: [
        { id: "n1", label: "JSX & Props", level: "beginner", parents: [] },
        { id: "n2", label: "Hooks & State", level: "intermediate", parents: ["n1"], isCurrent: true },
        { id: "n3", label: "Context & Reducers", level: "intermediate", parents: ["n2"] },
        { id: "n4", label: "Concurrent Mode", level: "advanced", parents: ["n3"] },
      ]
    }
  },
  {
    _id: "demo-note-2",
    title: "Distributed Systems & CAP Theorem",
    content: "# Distributed Systems Fundamentals\n\nDesigning resilient, highly available architectures.\n\n### The CAP Theorem\nIn a distributed data store, you can only pick two of:\n1. **Consistency**: Every read receives the most recent write.\n2. **Availability**: Every request receives a non-error response.\n3. **Partition Tolerance**: System continues to operate despite network partitions.\n\n> In practice, partition tolerance is non-negotiable across physical networks.",
    tags: ["system-design", "backend", "architecture"],
    sourceType: "text",
    processingStatus: "processed",
    createdAt: new Date(Date.now() - 3600000).toISOString(),
    analysis: {
      score: 78,
      level: "intermediate",
      summary: "Good grasp of distributed trade-offs and CAP theorem constraints.",
      strengths: ["CAP theorem clarity", "Partition tolerance awareness"],
      gaps: ["Event-driven consensus (Raft/Paxos)", "Distributed transactions (2PC/Sagas)"],
      currentNode: "n2",
      nodes: [
        { id: "n1", label: "Client-Server Basics", level: "beginner", parents: [] },
        { id: "n2", label: "CAP Theorem", level: "intermediate", parents: ["n1"], isCurrent: true },
        { id: "n3", label: "Message Queues & Event Streaming", level: "intermediate", parents: ["n2"] },
        { id: "n4", label: "Consensus Algorithms", level: "advanced", parents: ["n3"] },
      ]
    }
  }
];

const parseTags = (tagsInput) => {
  if (Array.isArray(tagsInput)) return tagsInput.map((t) => String(t).trim()).filter(Boolean);
  if (typeof tagsInput === "string") return tagsInput.split(",").map((t) => t.trim()).filter(Boolean);
  return [];
};

const deleteFileIfExists = (filePath) => {
  if (!filePath) return;
  try { if (fs.existsSync(filePath)) fs.unlinkSync(filePath); } catch (e) { console.error(e.message); }
};

/**
 * Run analysis in background — yields execution to Node.js Event Loop using setImmediate()
 */
const runAnalysisAsync = (noteId, note) => {
  setImmediate(async () => {
    try {
      const analysis = await analyzeNote(note);
      if (analysis) {
        if (isDbConnected()) {
          await Note.findByIdAndUpdate(noteId, { analysis });
        } else {
          const idx = memoryNotes.findIndex(n => n._id === noteId);
          if (idx !== -1) memoryNotes[idx].analysis = analysis;
        }
        console.log(`Analysis completed for note ${noteId}`);
      }
    } catch (err) {
      console.error(`Background analysis error for note ${noteId}:`, err.message);
    }
  });
};

/**
 * CREATE TEXT NOTE
 */
exports.createNote = async (req, res) => {
  try {
    const { title, content, tags = [] } = req.body;
    const noteData = {
      title: String(title || "").trim(),
      content: String(content || "").trim(),
      tags: parseTags(tags),
      sourceType: "text",
      processingStatus: "processed",
      createdAt: new Date().toISOString(),
    };

    if (isDbConnected()) {
      const note = new Note(noteData);
      const saved = await note.save();
      res.status(201).json(saved);
      runAnalysisAsync(saved._id, saved);
    } else {
      const newId = `note-${Date.now()}`;
      const saved = { _id: newId, ...noteData };
      memoryNotes.unshift(saved);
      res.status(201).json(saved);
      runAnalysisAsync(newId, saved);
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/**
 * CREATE FROM IMAGE
 * @concept File upload handling
 * @concept Server-side error handling
 * @concept HTTP status codes used correctly (400 Bad Request, 201 Created, 500 Error)
 */
exports.createImageNote = async (req, res) => {
  const uploadedFile = req.file;
  try {
    if (!uploadedFile) return res.status(400).json({ error: "image is required" });

    const { title, content, tags = [] } = req.body;
    const parsedTags = parseTags(tags);
    const imagePath = path.posix.join("uploads", uploadedFile.filename);
    const imageUrl = `/${imagePath}`;
    const noteTitle = String(title || "").trim() || path.parse(uploadedFile.originalname).name || "Untitled Note";

    let ocrText = "", ocrBlocks = [], ocrConfidence = null, processingStatus = "needs_review";
    try {
      const ocrResult = await extractTextFromImage(uploadedFile.path);
      ocrText = ocrResult.text || "";
      ocrBlocks = ocrResult.blocks || [];
      ocrConfidence = ocrResult.confidence;
      processingStatus = ocrText ? "processed" : "needs_review";
    } catch (ocrError) {
      console.error("OCR error:", ocrError.message);
    }

    const note = new Note({
      title: noteTitle,
      content: String(content || ocrText || "").trim(),
      tags: parsedTags,
      sourceType: "image",
      imagePath, imageUrl,
      originalFilename: uploadedFile.originalname || "",
      processingStatus, ocrText, ocrBlocks, ocrConfidence,
    });
    const saved = await note.save();
    res.status(201).json(saved);
    runAnalysisAsync(saved._id, saved);
  } catch (err) {
    deleteFileIfExists(uploadedFile?.path);
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET ALL NOTES
 */
exports.getAllNotes = async (req, res) => {
  try {
    const { tag, q } = req.query;

    if (isDbConnected()) {
      const filter = {};
      if (tag) filter.tags = tag;
      if (q) filter.$or = [
        { title: { $regex: q, $options: "i" } },
        { content: { $regex: q, $options: "i" } },
        { ocrText: { $regex: q, $options: "i" } },
      ];
      const notes = await Note.find(filter).sort({ createdAt: -1 });
      return res.json(notes);
    }

    // In-memory fallback
    let filtered = [...memoryNotes];
    if (tag) {
      filtered = filtered.filter(n => (n.tags || []).includes(tag));
    }
    if (q) {
      const qLower = q.toLowerCase();
      filtered = filtered.filter(n => 
        (n.title && n.title.toLowerCase().includes(qLower)) ||
        (n.content && n.content.toLowerCase().includes(qLower)) ||
        (n.ocrText && n.ocrText.toLowerCase().includes(qLower))
      );
    }
    res.json(filtered);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET NOTE BY ID
 */
exports.getNoteById = async (req, res) => {
  try {
    if (isDbConnected()) {
      const note = await Note.findById(req.params.id);
      if (!note) return res.status(404).json({ error: "Note not found" });
      return res.json(note);
    }

    const note = memoryNotes.find(n => n._id === req.params.id);
    if (!note) return res.status(404).json({ error: "Note not found" });
    res.json(note);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * UPDATE NOTE
 */
exports.updateNote = async (req, res) => {
  try {
    const { title, content, tags } = req.body;
    if (isDbConnected()) {
      const updated = await Note.findByIdAndUpdate(
        req.params.id,
        { title, content, tags: parseTags(tags) },
        { new: true }
      );
      if (!updated) return res.status(404).json({ error: "Note not found" });
      return res.json(updated);
    }

    const idx = memoryNotes.findIndex(n => n._id === req.params.id);
    if (idx === -1) return res.status(404).json({ error: "Note not found" });
    const existing = memoryNotes[idx];
    const updated = {
      ...existing,
      title: title !== undefined ? title : existing.title,
      content: content !== undefined ? content : existing.content,
      tags: tags !== undefined ? parseTags(tags) : existing.tags,
    };
    memoryNotes[idx] = updated;
    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE NOTE
 */
exports.deleteNote = async (req, res) => {
  try {
    if (isDbConnected()) {
      const note = await Note.findById(req.params.id);
      if (!note) return res.status(404).json({ error: "Note not found" });
      if (note.imagePath) deleteFileIfExists(path.join(__dirname, "..", note.imagePath));
      await Chat.deleteMany({ noteId: note._id });
      await Note.findByIdAndDelete(req.params.id);
      return res.json({ message: "Note deleted successfully" });
    }

    const idx = memoryNotes.findIndex(n => n._id === req.params.id);
    if (idx !== -1) memoryNotes.splice(idx, 1);
    res.json({ message: "Note deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * RE-ANALYZE NOTE
 */
exports.reAnalyzeNote = async (req, res) => {
  try {
    let note;
    if (isDbConnected()) {
      note = await Note.findById(req.params.id);
    } else {
      note = memoryNotes.find(n => n._id === req.params.id);
    }

    if (!note) return res.status(404).json({ error: "Note not found" });
    const analysis = await analyzeNote(note);
    if (!analysis) {
      return res.status(500).json({ error: "Failed to generate analysis" });
    }

    if (isDbConnected()) {
      const updated = await Note.findByIdAndUpdate(req.params.id, { analysis }, { new: true });
      return res.json({ analysis: updated ? updated.analysis : analysis });
    }

    const idx = memoryNotes.findIndex(n => n._id === req.params.id);
    if (idx !== -1) memoryNotes[idx].analysis = analysis;
    res.json({ analysis });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

exports.getNoteByIdInternal = (id) => {
  return memoryNotes.find(n => n._id === String(id)) || null;
};


