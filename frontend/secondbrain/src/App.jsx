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
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-dot" aria-hidden="true" />
          <div>
            <div className="brand">Second Brain</div>
            <p className="brand-subtitle">Intelligent Note Synthesis & Chat</p>
          </div>
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
