import { useState, useRef, useEffect } from "react";
import Home from "./pages/Home";
import ChatPage from "./pages/ChatPage";
import { animatePageTransition } from "./utils/animations";

function App() {
  const [activePage, setActivePage]     = useState("notes");
  const [selectedNote, setSelectedNote] = useState(null);
  const [prefillTopic, setPrefillTopic] = useState("");
  const pageWrapRef                     = useRef(null);

  const openChat = () => {
    setPrefillTopic("");
    setActivePage("chat");
  };

  // Called from flowchart node click: switches to chat with a pre-filled question
  const openChatWithTopic = (topicLabel, note) => {
    setSelectedNote(note);
    setPrefillTopic(`Explain "${topicLabel}" to me in the context of ${note.title}`);
    setActivePage("chat");
  };

  // Trigger Anime.js transition on tab/page change
  useEffect(() => {
    if (pageWrapRef.current) {
      animatePageTransition(pageWrapRef.current);
    }
  }, [activePage]);

  return (
    <div className="app-shell">
      {/* Subtle shape background pattern & ambient geometric wireframes */}
      <div className="bg-shapes-overlay" aria-hidden="true">
        <div className="shape-pattern" />
        <div className="shape-ambient-glow shape-glow-1" />
        <div className="shape-ambient-glow shape-glow-2" />

        {/* Ambient subtle geometric wireframe shapes for visual impact */}
        <svg className="shape-geo shape-geo-1" width="380" height="380" viewBox="0 0 380 380" fill="none">
          <polygon points="190,20 345,110 345,270 190,360 35,270 35,110" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1.5" strokeDasharray="6 6" />
          <polygon points="190,60 305,125 305,255 190,320 75,255 75,125" stroke="rgba(99, 102, 241, 0.14)" strokeWidth="1" />
          <circle cx="190" cy="190" r="75" stroke="rgba(167, 139, 250, 0.10)" strokeWidth="1" strokeDasharray="3 5" />
          <circle cx="190" cy="190" r="4" fill="rgba(139, 92, 246, 0.28)" />
          <line x1="190" y1="20" x2="190" y2="60" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1" />
          <line x1="345" y1="110" x2="305" y2="125" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1" />
          <line x1="345" y1="270" x2="305" y2="255" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1" />
          <line x1="190" y1="360" x2="190" y2="320" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1" />
          <line x1="35" y1="270" x2="75" y2="255" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1" />
          <line x1="35" y1="110" x2="75" y2="125" stroke="rgba(139, 92, 246, 0.12)" strokeWidth="1" />
        </svg>

        <svg className="shape-geo shape-geo-2" width="300" height="300" viewBox="0 0 300 300" fill="none">
          <circle cx="150" cy="150" r="130" stroke="rgba(139, 92, 246, 0.09)" strokeWidth="1.5" strokeDasharray="5 7" />
          <circle cx="150" cy="150" r="95" stroke="rgba(99, 102, 241, 0.12)" strokeWidth="1" />
          <rect x="80" y="80" width="140" height="140" transform="rotate(45 150 150)" stroke="rgba(168, 85, 247, 0.08)" strokeWidth="1" />
          <circle cx="150" cy="20" r="3" fill="rgba(139, 92, 246, 0.25)" />
          <circle cx="150" cy="280" r="3" fill="rgba(139, 92, 246, 0.25)" />
          <circle cx="20" cy="150" r="3" fill="rgba(99, 102, 241, 0.25)" />
          <circle cx="280" cy="150" r="3" fill="rgba(99, 102, 241, 0.25)" />
        </svg>

        <svg className="shape-geo shape-geo-3" width="260" height="260" viewBox="0 0 260 260" fill="none">
          <polygon points="130,25 240,215 20,215" stroke="rgba(139, 92, 246, 0.08)" strokeWidth="1.5" strokeDasharray="6 6" />
          <polygon points="130,75 205,200 55,200" stroke="rgba(99, 102, 241, 0.10)" strokeWidth="1" />
          <circle cx="130" cy="150" r="45" stroke="rgba(167, 139, 250, 0.08)" strokeWidth="1" />
          <circle cx="130" cy="150" r="3" fill="rgba(139, 92, 246, 0.22)" />
        </svg>
      </div>

      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-dot" aria-hidden="true" />
          <h2 className="brand-title">Intelligent Note Synthesis &amp; Chat</h2>
        </div>
        <nav className="nav" aria-label="Main navigation">
          <button
            className={activePage === "notes" ? "nav-pill active" : "nav-pill"}
            onClick={() => setActivePage("notes")}
          >
            Notes
          </button>
          <button
            className={activePage === "chat" ? "nav-pill active" : "nav-pill"}
            onClick={() => setActivePage("chat")}
          >
            Chat
          </button>
        </nav>
      </header>

      <main ref={pageWrapRef} className="page-container">
        {activePage === "notes" ? (
          <Home
            selectedNote={selectedNote}
            setSelectedNote={setSelectedNote}
            openChat={openChat}
            openChatWithTopic={openChatWithTopic}
          />
        ) : (
          <ChatPage
            selectedNote={selectedNote}
            prefillTopic={prefillTopic}
            onPrefillUsed={() => setPrefillTopic("")}
          />
        )}
      </main>
    </div>
  );
}

export default App;
