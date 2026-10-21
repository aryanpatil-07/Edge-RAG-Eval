import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const NoteDetail = ({
  note,
  imageSrc,
  draft,
  draftMode,
  actionLoading,
  onRunAction,
  onApplyDraft,
  onDelete,
  onOpenChat,
  onOpenAnalysis,
}) => {
  const draftRef = useRef(null);

  useEffect(() => {
    if (draft && draftRef.current) {
      draftRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [draft]);

  if (!note) {
    return (
      <div className="note-detail empty-state">
        <div className="state-card">Select a note to view details and AI tools.</div>
      </div>
    );
  }

  const content = String(note.content || note.ocrText || "").trim();
  const hasAnalysis = !!note.analysis;

  return (
    <section className="note-detail">
      <div className="note-detail-header">
        <div>
          <p className="detail-kicker">Selected note</p>
          <h2>{note.title}</h2>
          <p className="detail-meta">
            {note.sourceType === "image" ? "📎 Image note" : "📝 Text note"}
            {" · "}{note.processingStatus || "processed"}
            {hasAnalysis && (
              <span className={`analysis-badge level-${note.analysis.level}`}>
                {note.analysis.level} · {note.analysis.score}/100
              </span>
            )}
          </p>
        </div>

        <div className="detail-actions">
          <button
            type="button"
            className="analysis-trigger-btn"
            onClick={onOpenAnalysis}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight:6}}>
              <path d="M2 20h20M6 20V10M12 20V4M18 20v-6"/>
            </svg>
            {hasAnalysis ? "View Analysis" : "Analyse Knowledge"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => onRunAction("convert")}
            disabled={actionLoading !== ""}
          >
            {actionLoading === "convert" ? "Converting…" : "Convert format"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => onRunAction("improve")}
            disabled={actionLoading !== ""}
          >
            {actionLoading === "improve" ? "Improving…" : "Improve with AI"}
          </button>
          <button type="button" className="primary-button" onClick={onOpenChat}>
            Chat about note
          </button>
        </div>
      </div>

      <div className="note-detail-body">
        {imageSrc && (
          <div className="note-visual-panel">
            <div className="panel-label">Original image</div>
            <img className="note-image" src={imageSrc} alt={note.title} />
          </div>
        )}

        <div className="note-text-panel">
          <div className="panel-label">Note content</div>
          <div 
            className="note-markdown-body"
            style={{
              lineHeight: 1.8,
              fontSize: "0.96rem",
            }}
          >
            {content
              ? <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({node: _node, ...props}) => <h1 style={{marginTop: "1.4em", marginBottom: "0.6em", fontSize: "1.55rem", fontWeight: 700, color: "#eceef2"}} {...props} />,
                    h2: ({node: _node, ...props}) => <h2 style={{marginTop: "1.3em", marginBottom: "0.5em", fontSize: "1.3rem", fontWeight: 700, color: "#eceef2"}} {...props} />,
                    h3: ({node: _node, ...props}) => <h3 style={{marginTop: "1.2em", marginBottom: "0.4em", fontSize: "1.1rem", fontWeight: 700, color: "#eceef2"}} {...props} />,
                    p: ({node: _node, ...props}) => <p style={{margin: "0.8em 0", lineHeight: 1.8}} {...props} />,
                    ul: ({node: _node, ...props}) => <ul style={{paddingLeft: "1.8em", margin: "0.8em 0 1em"}} {...props} />,
                    ol: ({node: _node, ...props}) => <ol style={{paddingLeft: "1.8em", margin: "0.8em 0 1em"}} {...props} />,
                    li: ({node: _node, ...props}) => <li style={{margin: "0.5em 0", lineHeight: "1.7"}} {...props} />,
                    blockquote: ({node: _node, ...props}) => <blockquote style={{borderLeft: "4px solid #8b5cf6", margin: "1.2em 0", padding: "12px 0 12px 18px", fontStyle: "italic", background: "rgba(139,92,246,0.05)", borderRadius: "0 6px 6px 0", color: "#7a8494"}} {...props} />,
                    code: ({node: _node, ...props}) => <code style={{background: "rgba(139,92,246,0.13)", border: "1px solid rgba(139,92,246,0.2)", color: "#c4b5fd", padding: "3px 8px", borderRadius: "5px", fontSize: "0.88em"}} {...props} />,
                    pre: ({node: _node, ...props}) => <pre style={{background: "#1c2029", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", padding: "16px 18px", overflow: "auto", margin: "1.2em 0"}} {...props} />,
                    hr: ({node: _node, ...props}) => <hr style={{border: "none", borderTop: "1px solid rgba(255,255,255,0.07)", margin: "1.6em 0"}} {...props} />,
                  }}
                >
                  {content}
                </ReactMarkdown>
              : <p className="note-text-copy">No text extracted yet.</p>
            }
          </div>

          {(note.tags || []).length > 0 && (
            <div className="tag-list">
              {note.tags.map((tag) => (
                <span key={tag} className="tag-pill">{tag}</span>
              ))}
            </div>
          )}

          <div className="note-toolbar">
            <button type="button" className="danger-button" onClick={() => onDelete(note._id)}>
              🗑 Delete note
            </button>
          </div>
        </div>
      </div>

      {draft && (
        <div className="draft-panel" ref={draftRef} style={{ animation: "fadeIn 0.25s ease" }}>
          <div className="panel-label">
            AI draft · {draftMode === "convert" ? "Study Format" : "AI Improved"}
          </div>
          <div 
            className="note-markdown-body draft-markdown"
            style={{
              lineHeight: 1.8,
              fontSize: "0.96rem",
            }}
          >
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({node: _node, ...props}) => <h1 style={{marginTop: "1.4em", marginBottom: "0.6em", fontSize: "1.55rem", fontWeight: 700, color: "#eceef2"}} {...props} />,
                h2: ({node: _node, ...props}) => <h2 style={{marginTop: "1.3em", marginBottom: "0.5em", fontSize: "1.3rem", fontWeight: 700, color: "#eceef2"}} {...props} />,
                h3: ({node: _node, ...props}) => <h3 style={{marginTop: "1.2em", marginBottom: "0.4em", fontSize: "1.1rem", fontWeight: 700, color: "#eceef2"}} {...props} />,
                p: ({node: _node, ...props}) => <p style={{margin: "0.8em 0", lineHeight: 1.8}} {...props} />,
                ul: ({node: _node, ...props}) => <ul style={{paddingLeft: "1.8em", margin: "0.8em 0 1em"}} {...props} />,
                ol: ({node: _node, ...props}) => <ol style={{paddingLeft: "1.8em", margin: "0.8em 0 1em"}} {...props} />,
                li: ({node: _node, ...props}) => <li style={{margin: "0.5em 0", lineHeight: "1.7"}} {...props} />,
                blockquote: ({node: _node, ...props}) => <blockquote style={{borderLeft: "4px solid #8b5cf6", margin: "1.2em 0", padding: "12px 0 12px 18px", fontStyle: "italic", background: "rgba(139,92,246,0.05)", borderRadius: "0 6px 6px 0", color: "#7a8494"}} {...props} />,
                code: ({node: _node, ...props}) => <code style={{background: "rgba(139,92,246,0.13)", border: "1px solid rgba(139,92,246,0.2)", color: "#c4b5fd", padding: "3px 8px", borderRadius: "5px", fontSize: "0.88em"}} {...props} />,
                pre: ({node: _node, ...props}) => <pre style={{background: "#1c2029", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", padding: "16px 18px", overflow: "auto", margin: "1.2em 0"}} {...props} />,
                hr: ({node: _node, ...props}) => <hr style={{border: "none", borderTop: "1px solid rgba(255,255,255,0.07)", margin: "1.6em 0"}} {...props} />,
              }}
            >
              {draft}
            </ReactMarkdown>
          </div>
          <div className="detail-actions" style={{ marginTop: 14, display: "flex", gap: "10px" }}>
            <button type="button" className="primary-button" onClick={() => onApplyDraft(draft)}>
              Use this version
            </button>
            <button 
              type="button" 
              className="secondary-button" 
              onClick={() => onApplyDraft(null)}
              style={{ background: "rgba(255,255,255,0.05)" }}
            >
              Discard draft
            </button>
          </div>
        </div>
      )}
    </section>
  );
};

export default NoteDetail;
