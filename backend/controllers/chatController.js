const mongoose = require("mongoose");
const Chat = require("../models/Chat");
const Note = require("../models/Note");
const { askLLM, buildNotesContext, synthesizeTutorChat } = require("../services/aiService");
const { getRelevantNotes } = require("../services/retrievalService");

const isDbConnected = () => mongoose.connection.readyState === 1;
const memoryChats = new Map();

const createChat = async (req, res) => {
  try {
    const { noteId = null } = req.body || {};
    if (isDbConnected()) {
      const chat = await Chat.create({ messages: [], noteId: noteId || null });
      return res.status(201).json(chat);
    }

    const chatId = `chat-${Date.now()}`;
    const chat = { _id: chatId, messages: [], noteId: noteId || null };
    memoryChats.set(chatId, chat);
    res.status(201).json(chat);
  } catch (error) {
    console.error("Create chat error:", error.message);
    res.status(500).json({ error: error.message || "Failed to create chat" });
  }
};

const getChat = async (req, res) => {
  try {
    if (isDbConnected()) {
      const chat = await Chat.findById(req.params.chatId);
      if (!chat) return res.status(404).json({ error: "Chat not found" });
      return res.json(chat);
    }

    const chat = memoryChats.get(req.params.chatId);
    if (!chat) return res.status(404).json({ error: "Chat not found" });
    res.json(chat);
  } catch (error) {
    console.error("Get chat error:", error.message);
    res.status(500).json({ error: error.message || "Failed to load chat" });
  }
};

const sendMessage = async (req, res) => {
  try {
    const { chatId } = req.params;
    const { message, noteId } = req.body;

    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "message is required" });
    }

    let chat;
    if (isDbConnected()) {
      chat = await Chat.findById(chatId);
    } else {
      chat = memoryChats.get(chatId);
    }
    if (!chat) return res.status(404).json({ error: "Chat not found" });

    // Save user message
    chat.messages.push({ role: "user", content: message.trim() });

    // Recent history for context (last 6 messages)
    const historyText = chat.messages
      .slice(-6)
      .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
      .join("\n");

    // Resolve notes context + study subject label
    const selectedNoteId = noteId || chat.noteId || null;
    let notesContext = "";
    let studySubject = "your notes";

    if (selectedNoteId) {
      let selectedNote = null;
      if (isDbConnected()) {
        try {
          selectedNote = await Note.findById(selectedNoteId);
        } catch {
          selectedNote = null;
        }
      }
      if (selectedNote) {
        notesContext = buildNotesContext([selectedNote]);
        studySubject = selectedNote.title;
        chat.noteId = selectedNote._id;
      }
    }

    if (!notesContext && isDbConnected()) {
      try {
        const relevant = await getRelevantNotes(message);
        if (relevant.length > 0) {
          notesContext = buildNotesContext(relevant);
          studySubject = relevant.map((n) => n.title).join(", ");
        }
      } catch {
        // proceed without notes context
      }
    }

    // Build prompt — answers from general knowledge, notes used as context
    const prompt = `You are a knowledgeable tutor helping a learner who is studying: "${studySubject}".

${notesContext ? `Their personal notes on this subject:\n${notesContext}\n` : ""}
Conversation History:
${historyText || "No previous conversation."}

Current Question:
${message.trim()}

Instructions:
- Answer the question thoroughly using your expert general knowledge.
- Use the personal notes above as extra context to personalise your answer where relevant.
- For any concept or topic on a learning roadmap (e.g. "Explain X to me"), give a clear structured explanation with examples, analogies, and practical tips.
- Format with markdown: headings, bullet points, code blocks where appropriate.
- Be educational, clear, and encouraging. Never refuse a genuine learning question.`;

    let answer = "";
    try {
      answer = await askLLM(prompt);
    } catch (llmErr) {
      console.warn("Chat LLM error, using synthesized tutor response:", llmErr.message);
      answer = synthesizeTutorChat(prompt);
    }

    chat.messages.push({ role: "assistant", content: answer });
    if (typeof chat.save === "function") {
      await chat.save();
    } else {
      memoryChats.set(chatId, chat);
    }

    res.json({ answer, chatId: chat._id });
  } catch (error) {
    console.error("Send message error:", error.response?.data || error.message);
    res.status(500).json({
      error: error.response?.data?.error || error.message || "Failed to process message",
    });
  }
};

module.exports = { createChat, getChat, sendMessage };
