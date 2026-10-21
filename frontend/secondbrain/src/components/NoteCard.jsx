import { animateButtonPress } from "../utils/animations";

const cardTones = ["tone-coral", "tone-yellow", "tone-cream", "tone-mint"];

/** Returns the first non-empty line of text from markdown/plain content */
const getFirstLine = (text = "") => {
  const stripped = text
    .replace(/^#+\s*/gm, "")   // strip heading markers
    .replace(/\*\*|__|\*|_|~~|`/g, "") // strip inline markers
    .trim();
  const firstLine = stripped.split("\n").find((l) => l.trim() !== "");
  return firstLine?.trim() || "";
};

const NoteCard = ({ note, onDelete, onSelect, index = 0, active = false }) => {
  const toneClass = cardTones[index % cardTones.length];
  const preview = getFirstLine(note.content);

  const handleClick = (e) => {
    animateButtonPress(e.currentTarget);
    onSelect?.(note);
  };

  return (
    <div
      className={`note-card ${toneClass} ${active ? "selected" : ""}`}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && handleClick(e)}
    >
      <div className="note-card-header">
        <div className="note-card-titles">
          <h3>{note.title}</h3>
          <p className="note-card-subtitle">
            {note.sourceType === "image" ? "📎 Image note" : "📝 Text note"}
            {note.processingStatus ? ` · ${note.processingStatus}` : ""}
          </p>
        </div>
        {onDelete && (
          <button
            className="note-delete-btn"
            onClick={(e) => {
              e.stopPropagation();
              animateButtonPress(e.currentTarget);
              onDelete(note._id);
            }}
            type="button"
            aria-label="Delete note"
          >
            Delete
          </button>
        )}
      </div>

      {preview && <p className="note-card-copy">{preview}</p>}

      <div className="note-card-footer">
        {note.tags?.length > 0 && (
          <div className="tag-list">
            {note.tags.map((tag) => (
              <span key={tag} className="tag-pill">
                {tag}
              </span>
            ))}
          </div>
        )}

        <div className="note-card-badges">
          {note.analysis && (
            <span className={`analysis-pill-badge level-${note.analysis.level}`} title="Knowledge analysis completed">
              ✦ {note.analysis.score}/100
            </span>
          )}
          {note.imageUrl ? <span className="attachment-badge">📷 Image</span> : null}
        </div>
      </div>
    </div>
  );
};

export default NoteCard;