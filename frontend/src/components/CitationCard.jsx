import React, { useState } from 'react';
import { FileText, Copy, Check, ChevronDown, ChevronUp, Quote, ExternalLink } from 'lucide-react';

export default function CitationCard({ citation, index }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(citation.quoted_snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`citation-card-enhanced ${expanded ? 'expanded' : ''}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="citation-top-bar">
        <div className="citation-source-group">
          <div className="citation-icon-wrap">
            <FileText size={14} color="#38bdf8" />
          </div>
          <span className="citation-source-name" title={citation.source}>
            {citation.source}
          </span>
        </div>

        <div className="citation-meta-group">
          <span className="citation-page-badge">
            Page {citation.page}
          </span>
          <div className="citation-chevron">
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </div>
        </div>
      </div>

      <div className="citation-quote-box">
        <Quote size={12} className="quote-mark" />
        <span className="citation-quote-text">
          {citation.quoted_snippet}
        </span>
      </div>

      <div className="citation-bottom-meta">
        <span className="citation-chunk-pill">
          #{citation.chunk_id}
        </span>

        <button
          onClick={handleCopy}
          className="citation-copy-btn"
          title="Copy verified quote"
        >
          {copied ? <Check size={12} color="#34d399" /> : <Copy size={12} />}
          <span>{copied ? 'Copied' : 'Copy Quote'}</span>
        </button>
      </div>
    </div>
  );
}
