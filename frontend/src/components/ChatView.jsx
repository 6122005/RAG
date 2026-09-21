import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Sparkles,
  Bot,
  User,
  ShieldAlert,
  Clock,
  Copy,
  Check,
  RotateCcw,
  FileCheck2,
} from 'lucide-react';
import ConfidenceBadge from './ConfidenceBadge';
import CitationCard from './CitationCard';
import FormattedAnswer from './FormattedAnswer';

export default function ChatView({
  messages,
  onSendMessage,
  loading,
  conversationId,
  onResetSession,
}) {
  const [input, setInput] = useState('');
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});
  const messagesEndRef = useRef(null);

  const toggleSources = (idx) => {
    setExpandedSources((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const clean = input.trim();
    if (!clean || loading) return;
    onSendMessage(clean);
    setInput('');
  };

  const handleCopyAnswer = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <main className="main-chat">
      {/* Sleek Header */}
      <header className="chat-header">
        <div className="header-title-group">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="header-sparkle-wrap">
              <Sparkles size={18} color="#38bdf8" />
            </div>
            <div>
              <h1>Grounded Knowledge Assistant</h1>
              <p>Strictly citation-backed • Zero parametric hallucination</p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="session-id-pill" title="LangGraph Checkpointer Session Thread">
            <span className="session-dot" />
            <span>Session: {conversationId.slice(0, 8)}</span>
          </div>

          <button
            onClick={onResetSession}
            className="new-session-btn"
            title="Start fresh conversation"
          >
            <RotateCcw size={13} />
            <span>New Chat</span>
          </button>
        </div>
      </header>

      {/* Messages Scroll Area */}
      <div className="messages-container">
        {messages.map((msg, i) => (
          <div key={i} className={`message-row ${msg.role}`}>
              {msg.role === 'assistant' && (
                <div className="avatar bot">
                  <Bot size={18} />
                </div>
              )}

              <div className="bubble-enhanced">
                {/* Assistant Toolbar */}
                {msg.role === 'assistant' && (
                  <div className="assistant-meta-bar" style={{ justifyContent: 'flex-end' }}>
                    <button
                      onClick={() => handleCopyAnswer(msg.content, i)}
                      className="copy-answer-btn"
                      title="Copy response"
                    >
                      {copiedIndex === i ? (
                        <>
                          <Check size={13} color="#34d399" />
                          <span style={{ color: '#34d399' }}>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy size={13} />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                )}

                {/* Refusal Alert Visual State */}
                {msg.role === 'assistant' && msg.can_answer === false && (
                  <div className="refusal-banner-enhanced">
                    <div className="refusal-icon-wrap">
                      <ShieldAlert size={20} color="#f87171" />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div className="refusal-title-row">
                        <span className="refusal-pill">Zero Hallucination Refusal</span>
                        <span className="refusal-subtext">Guardrail Triggered</span>
                      </div>
                      <p className="refusal-desc">
                        The retrieved knowledge base does not contain verified documentation to answer
                        this query. To guarantee 100% correctness, generation was halted.
                      </p>
                    </div>
                  </div>
                )}

                {/* Formatted Answer Body */}
                <div className="answer-body-content">
                  <FormattedAnswer text={msg.content} />
                </div>

                {/* Grounding & Citations Collapsible Drawer */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations-wrapper-enhanced" style={{ marginTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '8px' }}>
                    <button
                      type="button"
                      onClick={() => toggleSources(i)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--text-muted)',
                        fontSize: '0.74rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '4px 0',
                        transition: 'color 0.15s',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent-cyan)')}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                    >
                      <FileCheck2 size={13} color="var(--accent-cyan)" />
                      <span>{expandedSources[i] ? 'Hide' : 'View'} Document Sources ({msg.citations.length})</span>
                    </button>

                    {expandedSources[i] && (
                      <div className="citation-grid" style={{ marginTop: '8px' }}>
                        {msg.citations.map((c, citIdx) => (
                          <CitationCard key={citIdx} citation={c} index={citIdx} />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div className="avatar user">
                  <User size={18} />
                </div>
              )}
            </div>
          ))}

        {/* Loading Bubble */}
        {loading && (
          <div className="message-row assistant">
            <div className="avatar bot pulsing">
              <Bot size={18} />
            </div>
            <div className="bubble-enhanced loading-bubble">
              <div className="loading-dots">
                <span />
                <span />
                <span />
              </div>
              <span className="loading-text">
                Executing hybrid retrieval, reranking passages with cross-encoder, and validating citations...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Modern Floating Input Area */}
      <div className="chat-input-area">
        <form onSubmit={handleSubmit} className="input-box-wrapper-enhanced">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question grounded strictly in your documents..."
            disabled={loading}
            autoFocus
          />
          <button
            type="submit"
            className="send-btn-enhanced"
            disabled={loading || !input.trim()}
            title="Send query"
          >
            <Send size={16} />
          </button>
        </form>
      </div>
    </main>
  );
}
