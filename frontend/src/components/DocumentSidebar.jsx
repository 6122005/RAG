import React, { useState, useRef } from 'react';
import {
  Files,
  UploadCloud,
  FileText,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  MessageSquare,
  Trash2,
  Plus,
} from 'lucide-react';
import { ingestFile } from '../api/client';

export default function DocumentSidebar({
  documents,
  loadingDocs,
  onRefreshDocs,
  health,
  apiKey,
  onApiKeyChange,
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}) {
  const [activeTab, setActiveTab] = useState('docs'); // 'docs' or 'sessions'
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(null);
    try {
      const res = await ingestFile(file, apiKey);
      setUploadStatus({
        type: 'success',
        message: 'PDF successfully uploaded',
      });
      onRefreshDocs();
    } catch (err) {
      setUploadStatus({
        type: 'error',
        message: err.message || 'Ingestion failed',
      });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <aside className="sidebar">
      {/* Header */}
      <div
        style={{
          padding: '18px 20px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))',
              borderRadius: '8px',
              padding: '6px',
              display: 'flex',
              color: '#04101e',
            }}
          >
            <HardDrive size={18} />
          </div>
          <div>
            <h2 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              RAG Workspace
            </h2>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              ChromaDB + LangGraph
            </span>
          </div>
        </div>

        <button
          onClick={onNewSession}
          style={{
            background: 'rgba(56, 189, 248, 0.12)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '6px',
            color: 'var(--accent-cyan)',
            padding: '5px 8px',
            fontSize: '0.72rem',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
          title="Start new chat session"
        >
          <Plus size={13} />
          <span>New</span>
        </button>
      </div>

      {/* Tab Switcher: Sources vs Sessions */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid var(--border-subtle)',
          background: 'rgba(0, 0, 0, 0.2)',
          padding: '4px 8px',
          gap: '6px',
        }}
      >
        <button
          onClick={() => setActiveTab('docs')}
          style={{
            flex: 1,
            padding: '7px 10px',
            border: 'none',
            borderRadius: '6px',
            background: activeTab === 'docs' ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
            color: activeTab === 'docs' ? 'var(--text-primary)' : 'var(--text-muted)',
            fontSize: '0.76rem',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'all 0.15s',
          }}
        >
          <Files size={13} />
          <span>Sources ({documents.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('sessions')}
          style={{
            flex: 1,
            padding: '7px 10px',
            border: 'none',
            borderRadius: '6px',
            background: activeTab === 'sessions' ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
            color: activeTab === 'sessions' ? 'var(--text-primary)' : 'var(--text-muted)',
            fontSize: '0.76rem',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'all 0.15s',
          }}
        >
          <MessageSquare size={13} />
          <span>History ({sessions.length})</span>
        </button>
      </div>

      {/* TAB 1: Documents & Upload */}
      {activeTab === 'docs' && (
        <>
          {/* File Upload Drop Area */}
          <div style={{ padding: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,.txt,.md,.markdown"
              style={{ display: 'none' }}
            />
            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '2px dashed var(--border-subtle)',
                borderRadius: '10px',
                padding: '16px 12px',
                textAlign: 'center',
                cursor: uploading ? 'wait' : 'pointer',
                background: 'rgba(255, 255, 255, 0.02)',
                transition: 'border-color 0.2s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent-cyan)')}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
            >
              <UploadCloud
                size={22}
                color="var(--accent-cyan)"
                style={{ margin: '0 auto 6px', display: 'block' }}
              />
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {uploading ? 'Chunking & Indexing...' : 'Upload Document'}
              </span>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                PDF, Markdown, or TXT
              </p>
            </div>

            {uploadStatus && (
              <div
                style={{
                  marginTop: '10px',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  fontSize: '0.74rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background:
                    uploadStatus.type === 'success'
                      ? 'rgba(16, 185, 129, 0.1)'
                      : 'rgba(239, 68, 68, 0.1)',
                  color:
                    uploadStatus.type === 'success'
                      ? '#34d399'
                      : '#f87171',
                  border: `1px solid ${
                    uploadStatus.type === 'success'
                      ? 'rgba(16, 185, 129, 0.3)'
                      : 'rgba(239, 68, 68, 0.3)'
                  }`,
                }}
              >
                {uploadStatus.type === 'success' ? (
                  <CheckCircle2 size={13} />
                ) : (
                  <AlertCircle size={13} />
                )}
                <span>{uploadStatus.message}</span>
              </div>
            )}
          </div>

          {/* Sources List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
            <div
              style={{
                fontSize: '0.74rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: 'var(--text-muted)',
                marginBottom: '12px',
                display: 'flex',
                justifyContent: 'space-between',
              }}
            >
              <span>Ingested Files</span>
              <button
                onClick={onRefreshDocs}
                disabled={loadingDocs}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                }}
              >
                <RefreshCw size={12} className={loadingDocs ? 'spin-icon' : ''} />
              </button>
            </div>

            {documents.length === 0 ? (
              <div
                style={{
                  textAlign: 'center',
                  padding: '24px 12px',
                  color: 'var(--text-muted)',
                  fontSize: '0.8rem',
                }}
              >
                <Files size={24} style={{ opacity: 0.3, margin: '0 auto 8px', display: 'block' }} />
                No documents uploaded yet.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {documents.map((doc, i) => (
                  <div
                    key={i}
                    style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      padding: '10px 12px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '10px',
                    }}
                  >
                    <FileText size={16} color="var(--accent-cyan)" style={{ marginTop: '2px' }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: '0.82rem',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                        title={doc.filename}
                      >
                        {doc.filename}
                      </div>
                      <div
                        style={{
                          fontSize: '0.72rem',
                          color: 'var(--text-muted)',
                          display: 'flex',
                          gap: '8px',
                          marginTop: '3px',
                        }}
                      >
                        <span>{doc.chunk_count} chunks</span>
                        <span>•</span>
                        <span>{doc.total_pages} {doc.total_pages === 1 ? 'page' : 'pages'}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* TAB 2: Chat Sessions History */}
      {activeTab === 'sessions' && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
          <div
            style={{
              fontSize: '0.74rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--text-muted)',
              marginBottom: '12px',
            }}
          >
            <span>Previous Conversations</span>
          </div>

          {sessions.length === 0 ? (
            <div
              style={{
                textAlign: 'center',
                padding: '28px 12px',
                color: 'var(--text-muted)',
                fontSize: '0.8rem',
              }}
            >
              <MessageSquare size={24} style={{ opacity: 0.3, margin: '0 auto 8px', display: 'block' }} />
              No chat history yet. Ask a question to start one!
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {sessions.map((s) => {
                const isActive = s.id === activeSessionId;
                return (
                  <div
                    key={s.id}
                    onClick={() => onSelectSession(s.id)}
                    style={{
                      background: isActive
                        ? 'rgba(56, 189, 248, 0.12)'
                        : 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${
                        isActive ? 'var(--accent-cyan)' : 'var(--border-subtle)'
                      }`,
                      borderRadius: '8px',
                      padding: '10px 12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      transition: 'all 0.15s',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
                      <div
                        style={{
                          width: '6px',
                          height: '6px',
                          borderRadius: '50%',
                          background: isActive ? '#38bdf8' : 'transparent',
                          border: `1px solid ${isActive ? '#38bdf8' : 'var(--text-muted)'}`,
                          flexShrink: 0,
                        }}
                      />
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div
                          style={{
                            fontSize: '0.82rem',
                            fontWeight: isActive ? 700 : 500,
                            color: isActive ? 'var(--text-primary)' : '#cbd5e1',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                          title={s.title}
                        >
                          {s.title || 'Conversation'}
                        </div>
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                          {s.messages ? s.messages.length : 0} messages
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={(e) => onDeleteSession(s.id, e)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--text-muted)',
                        padding: '4px',
                        cursor: 'pointer',
                        opacity: 0.6,
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = '#f87171')}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                      title="Delete session"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
