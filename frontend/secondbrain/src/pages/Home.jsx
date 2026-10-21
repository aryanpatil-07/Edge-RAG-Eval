/**
 * @file Home.jsx
 * @concept React component composition
 * @concept Side effects with useEffect
 * @concept State management with useState
 * @concept Async data fetching from API
 * @concept JavaScript — Closures (cancelled boolean flag in useEffect cleanup)
 */

import { useEffect, useRef, useState } from "react";
import SearchBar from "../components/SearchBar";
import TagFilter from "../components/TagFilter";
import NoteCard from "../components/NoteCard";
import AddNote from "../components/AddNote";
import NoteDetail from "../components/NoteDetail";
import KnowledgeAnalysis from "../components/KnowledgeAnalysis";
import { API_BASE_URL, askNoteAction, deleteNote, getNote, getNotes, updateNote, reAnalyzeNote } from "../services/api";
import { animateCardsEntrance, animateModalEnter, animateButtonPress } from "../utils/animations";

const Home = ({ selectedNote, setSelectedNote, openChat, openChatWithTopic }) => {
  const [searchQuery, setSearchQuery]     = useState("");
  const [selectedTag, setSelectedTag]     = useState("");
  const [notes, setNotes]                 = useState([]);
  const [loading, setLoading]             = useState(true);
  const [draft, setDraft]                 = useState("");
  const [draftMode, setDraftMode]         = useState("");
  const [actionLoading, setActionLoading] = useState("");
  const [showAddNote, setShowAddNote]     = useState(false);
  const [showAnalysis, setShowAnalysis]   = useState(false);

  const detailBackdropRef = useRef(null);
  const detailModalRef    = useRef(null);

  /**
   * Side Effect Hook monitoring searchQuery and selectedTag
   */
  useEffect(() => {
    let cancelled = false;
    const loadNotes = async () => {
      try {
        setLoading(true);
        const res = await getNotes({ q: searchQuery || undefined, tag: selectedTag || undefined });
        if (!cancelled) setNotes(Array.isArray(res) ? res : []);
      } catch {
        if (!cancelled) setNotes([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadNotes();
    return () => { cancelled = true; };
  }, [searchQuery, selectedTag]);

  // Trigger Anime.js staggered card entrance whenever notes load/change
  useEffect(() => {
    if (!loading && notes.length > 0) {
      // Allow DOM repaint before measuring targets
      const timer = setTimeout(() => {
        animateCardsEntrance(".note-card");
      }, 30);
      return () => clearTimeout(timer);
    }
  }, [loading, notes.length, searchQuery, selectedTag]);

  // Trigger Anime.js modal entrance on detail modal open
  useEffect(() => {
    if (selectedNote && !showAnalysis && detailModalRef.current) {
      animateModalEnter(detailBackdropRef.current, detailModalRef.current);
    }
  }, [selectedNote, showAnalysis]);

  const noteTags = [...new Set(notes.flatMap((n) => n.tags || []))];

  const handleCreated = (newNote) => {
    setNotes((prev) => [newNote, ...prev]);
    setSelectedNote(newNote);
    setDraft(""); setDraftMode("");
    setShowAddNote(false);
  };

  const handleDelete = async (id) => {
    await deleteNote(id);
    setNotes((prev) => prev.filter((n) => n._id !== id));
    if (selectedNote?._id === id) {
      setSelectedNote(null); setDraft(""); setDraftMode("");
      setShowAnalysis(false);
    }
  };

  const handleSelect = async (note) => {
    setDraft(""); setDraftMode("");
    setShowAnalysis(false);
    try {
      const fresh = await getNote(note._id);
      setSelectedNote(fresh);
    } catch {
      setSelectedNote(note);
    }
  };

  const handleCloseDetail = () => {
    setSelectedNote(null); setDraft(""); setDraftMode("");
  };

  const handleRunAction = async (mode) => {
    if (!selectedNote?._id) return;
    setActionLoading(mode); setDraft(""); setDraftMode(mode);
    try {
      const res = await askNoteAction({ noteId: selectedNote._id, mode });
      if (res?.draft) {
        setDraft(res.draft);
      }
    } catch (err) {
      console.error("Action error:", err);
    } finally {
      setActionLoading("");
    }
  };

  const handleApplyDraft = async (draftText) => {
    if (!selectedNote?._id) return;
    try {
      const updated = await updateNote(selectedNote._id, {
        title: selectedNote.title, content: draftText, tags: selectedNote.tags || [],
      });
      setNotes((prev) => prev.map((n) => (n._id === updated._id ? updated : n)));
      setSelectedNote(updated); 
      setDraft(""); 
      setDraftMode("");
      
      try {
        const reAnalyzed = await reAnalyzeNote(selectedNote._id);
        if (reAnalyzed?.analysis) {
          const updatedWithAnalysis = { ...updated, analysis: reAnalyzed.analysis };
          setSelectedNote(updatedWithAnalysis);
          setNotes((prev) => prev.map((n) => (n._id === updated._id ? updatedWithAnalysis : n)));
        }
      } catch (error) {
        console.error("Re-analysis error:", error);
      }
    } catch (err) {
      console.error("Apply draft error:", err);
    }
  };

  const handleTopicChat = (topicLabel, note) => {
    setShowAnalysis(false);
    setSelectedNote(note);
    openChatWithTopic(topicLabel, note);
  };

  return (
    <div className="page">
      <div className="hero hero-row">
        <div>
          <h1>Second Brain</h1>
          <p>Your intelligent note workspace with AI synthesis, analysis & chat.</p>
        </div>
        <div className="hero-actions">
          <div className="hero-stats-badge" title="Total saved notes">
            <span className="stats-count">{notes.length}</span>
            <span className="stats-label">{notes.length === 1 ? "note" : "notes"}</span>
          </div>
          <button 
            className="add-note-fab" 
            onClick={(e) => {
              animateButtonPress(e.currentTarget);
              setShowAddNote(true);
            }} 
            aria-label="Add new note"
            title="Create new note"
          >
            <span>+</span>
          </button>
        </div>
      </div>

      <div className="toolbar">
        <SearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
        <TagFilter selectedTag={selectedTag} setSelectedTag={setSelectedTag} noteTags={noteTags} />
      </div>

      {loading ? (
        <div className="state-card state-loading">
          <div className="state-spinner" />
          <span>Loading knowledge vault…</span>
        </div>
      ) : !notes.length ? (
        <div className="state-card empty-vault">
          <div className="empty-vault-icon">✦</div>
          <h3>Your Second Brain is ready</h3>
          <p>
            {searchQuery || selectedTag 
              ? "No notes matched your current search filters." 
              : "Capture your first idea, meeting summary, or diagram."}
          </p>
          <button
            type="button"
            className="primary-button"
            style={{ marginTop: 14 }}
            onClick={(e) => {
              animateButtonPress(e.currentTarget);
              setShowAddNote(true);
            }}
          >
            + Create your first note
          </button>
        </div>
      ) : (
        <div className="notes-grid">
          {notes.map((note, index) => (
            <NoteCard
              key={note._id}
              note={note}
              index={index}
              onDelete={handleDelete}
              onSelect={handleSelect}
              active={selectedNote?._id === note._id}
            />
          ))}
        </div>
      )}

      {showAddNote && (
        <AddNote onCreated={handleCreated} onClose={() => setShowAddNote(false)} />
      )}

      {selectedNote && !showAnalysis && (
        <div 
          ref={detailBackdropRef}
          className="modal-backdrop" 
          onClick={(e) => e.target === e.currentTarget && handleCloseDetail()}
        >
          <div ref={detailModalRef} className="detail-modal">
            <button className="modal-close detail-modal-close" onClick={handleCloseDetail} aria-label="Close">×</button>
            <NoteDetail
              note={selectedNote}
              imageSrc={selectedNote?.imageUrl ? `${API_BASE_URL}${selectedNote.imageUrl}` : ""}
              draft={draft}
              draftMode={draftMode}
              actionLoading={actionLoading}
              onRunAction={handleRunAction}
              onApplyDraft={handleApplyDraft}
              onDelete={handleDelete}
              onOpenChat={openChat}
              onOpenAnalysis={() => setShowAnalysis(true)}
            />
          </div>
        </div>
      )}

      {selectedNote && showAnalysis && (
        <KnowledgeAnalysis
          note={selectedNote}
          onClose={() => setShowAnalysis(false)}
          onTopicChat={handleTopicChat}
        />
      )}
    </div>
  );
};

export default Home;
