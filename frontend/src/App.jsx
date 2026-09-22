import React, { useState, useEffect } from 'react';
import DocumentSidebar from './components/DocumentSidebar';
import ChatView from './components/ChatView';
import { queryAPI, fetchDocuments, checkHealth, DEFAULT_API_KEY } from './api/client';

function loadStoredSessions() {
  try {
    const raw = localStorage.getItem('rag_all_sessions');
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((s) => s.messages && s.messages.length > 0)
      : [];
  } catch {
    return [];
  }
}

function saveStoredSessions(sessions) {
  try {
    const valid = sessions.filter((s) => s.messages && s.messages.length > 0);
    localStorage.setItem('rag_all_sessions', JSON.stringify(valid));
  } catch (err) {
    console.error('Failed to save sessions:', err);
  }
}

export default function App() {
  const [sessions, setSessions] = useState(loadStoredSessions);
  const [conversationId, setConversationId] = useState(() => {
    const stored = localStorage.getItem('rag_active_session_id');
    if (stored) return stored;
    const newId = crypto.randomUUID();
    localStorage.setItem('rag_active_session_id', newId);
    return newId;
  });

  const [apiKey, setApiKey] = useState(() => localStorage.getItem('rag_api_key') || DEFAULT_API_KEY);
  const [messages, setMessages] = useState(() => {
    const storedSessions = loadStoredSessions();
    const current = storedSessions.find((s) => s.id === localStorage.getItem('rag_active_session_id'));
    return current ? current.messages : [];
  });

  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [health, setHealth] = useState(null);

  // Sync messages to session store whenever messages change
  useEffect(() => {
    if (!conversationId) return;

    setSessions((prevSessions) => {
      const existingIdx = prevSessions.findIndex((s) => s.id === conversationId);

      // Do not store new sessions if they have zero messages
      if (messages.length === 0 && existingIdx < 0) {
        return prevSessions;
      }

      const firstUserMsg = messages.find((m) => m.role === 'user');
      const title = firstUserMsg
        ? firstUserMsg.content.slice(0, 40) + (firstUserMsg.content.length > 40 ? '...' : '')
        : existingIdx >= 0
        ? prevSessions[existingIdx].title
        : 'New Conversation';

      const updatedSession = {
        id: conversationId,
        title,
        updatedAt: Date.now(),
        messages,
      };

      let newSessions;
      if (existingIdx >= 0) {
        newSessions = [...prevSessions];
        newSessions[existingIdx] = updatedSession;
      } else {
        newSessions = [updatedSession, ...prevSessions];
      }

      saveStoredSessions(newSessions);
      return newSessions;
    });
  }, [messages, conversationId]);

  // Switch to an existing or old session
  const handleSelectSession = (sessionId) => {
    if (sessionId === conversationId) return;
    const target = sessions.find((s) => s.id === sessionId);
    if (target) {
      setConversationId(target.id);
      localStorage.setItem('rag_active_session_id', target.id);
      setMessages(target.messages || []);
    }
  };

  // Start fresh conversation session
  const handleResetSession = () => {
    // If active conversation already has 0 messages, do not change session
    if (messages.length === 0) return;

    const newId = crypto.randomUUID();
    setConversationId(newId);
    localStorage.setItem('rag_active_session_id', newId);
    setMessages([]);
  };

  // Delete a saved session
  const handleDeleteSession = (sessionId, e) => {
    e.stopPropagation();
    const filtered = sessions.filter((s) => s.id !== sessionId);
    setSessions(filtered);
    saveStoredSessions(filtered);

    if (sessionId === conversationId) {
      handleResetSession();
    }
  };

  // Save API key changes
  const handleApiKeyChange = (newKey) => {
    setApiKey(newKey);
    localStorage.setItem('rag_api_key', newKey);
  };

  const loadDocuments = async () => {
    setLoadingDocs(true);
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoadingDocs(false);
    }
  };

  const loadHealth = async () => {
    try {
      const h = await checkHealth();
      setHealth(h);
    } catch (err) {
      setHealth({ status: 'unreachable' });
    }
  };

  useEffect(() => {
    loadDocuments();
    loadHealth();
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (text) => {
    const userMsg = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await queryAPI({
        query: text,
        conversationId,
        apiKey,
      });

      if (res.conversation_id && res.conversation_id !== conversationId) {
        setConversationId(res.conversation_id);
        localStorage.setItem('rag_active_session_id', res.conversation_id);
      }

      const botMsg = {
        role: 'assistant',
        content: res.answer,
        citations: res.citations || [],
        confidence: res.confidence || 'low',
        can_answer: res.can_answer,
        latency_ms: res.latency_ms,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const isNetworkErr =
        err.message &&
        (err.message.includes('fetch') ||
          err.message.includes('NetworkError') ||
          err.message.includes('502') ||
          err.message.includes('503'));
      const errorMsg = {
        role: 'assistant',
        content: isNetworkErr
          ? 'Connecting to backend... Free cloud instances go to sleep when idle. Please wait 10-15 seconds and try asking again.'
          : `Error: ${err.message || 'Unable to execute query.'}`,
        citations: [],
        confidence: 'low',
        can_answer: null,
        is_error: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <DocumentSidebar
        documents={documents}
        loadingDocs={loadingDocs}
        onRefreshDocs={loadDocuments}
        health={health}
        apiKey={apiKey}
        onApiKeyChange={handleApiKeyChange}
        sessions={sessions}
        activeSessionId={conversationId}
        onSelectSession={handleSelectSession}
        onNewSession={handleResetSession}
        onDeleteSession={handleDeleteSession}
      />
      <ChatView
        messages={messages}
        onSendMessage={handleSendMessage}
        loading={loading}
        conversationId={conversationId}
        onResetSession={handleResetSession}
      />
    </div>
  );
}
