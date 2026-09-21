import React from 'react';
import { CheckCircle, Sparkles, BookOpen, Quote } from 'lucide-react';

/**
 * Parses markdown-like text into structured, elegant React elements.
 * Handles headers, bold text, bullet points, numbered lists, and quotes.
 */
export default function FormattedAnswer({ text }) {
  if (!text) return null;

  // Split into paragraphs / lines
  const lines = text.split('\n');
  const elements = [];
  let currentList = [];
  let keyIndex = 0;

  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`ul-${keyIndex++}`} className="formatted-list">
          {currentList.map((item, idx) => (
            <li key={idx} className="formatted-list-item">
              <span className="list-bullet">✦</span>
              <span className="list-content">{renderInlineText(item)}</span>
            </li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  const renderInlineText = (str) => {
    // Replace **bold** with <strong>
    const parts = str.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={idx} className="bold-highlight">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      return;
    }

    // Defensive: filter out any legacy debug metadata lines
    if (
      trimmed.includes('Source Reference:') ||
      trimmed.includes('Chunk ID:') ||
      trimmed.includes('Verification: 100% grounded') ||
      trimmed === '### Verified Key Details'
    ) {
      return;
    }

    // Header 3 or 4
    if (trimmed.startsWith('### ') || trimmed.startsWith('#### ')) {
      flushList();
      const title = trimmed.replace(/^#{3,4}\s+/, '');
      elements.push(
        <div key={`h-${keyIndex++}`} className="formatted-heading">
          <Sparkles size={14} className="heading-icon" />
          <span>{title}</span>
        </div>
      );
    }
    // Bullet point
    else if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
      const itemText = trimmed.replace(/^[-*•]\s+/, '');
      currentList.push(itemText);
    }
    // Numbered item (e.g. 1. )
    else if (/^\d+\.\s+/.test(trimmed)) {
      const itemText = trimmed.replace(/^\d+\.\s+/, '');
      currentList.push(itemText);
    }
    // Standard paragraph
    else {
      flushList();
      elements.push(
        <p key={`p-${keyIndex++}`} className="formatted-paragraph">
          {renderInlineText(trimmed)}
        </p>
      );
    }
  });

  flushList();

  return <div className="formatted-answer-container">{elements}</div>;
}
