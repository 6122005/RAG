import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function ConfidenceBadge({ confidence }) {
  const conf = (confidence || 'low').toLowerCase();

  const configs = {
    high: {
      label: 'High Confidence',
      subtext: '100% Grounded in Sources',
      icon: ShieldCheck,
      color: '#34d399',
      border: 'rgba(16, 185, 129, 0.35)',
      bg: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08))',
      glow: '0 0 12px rgba(16, 185, 129, 0.25)',
      dotColor: '#10b981',
    },
    medium: {
      label: 'Medium Confidence',
      subtext: 'Partially Verified Context',
      icon: AlertTriangle,
      color: '#fbbf24',
      border: 'rgba(245, 158, 11, 0.35)',
      bg: 'linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.08))',
      glow: '0 0 12px rgba(245, 158, 11, 0.25)',
      dotColor: '#f59e0b',
    },
    low: {
      label: 'Low Confidence',
      subtext: 'Unverified / Insufficient Evidence',
      icon: ShieldAlert,
      color: '#f87171',
      border: 'rgba(239, 68, 68, 0.35)',
      bg: 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.08))',
      glow: '0 0 12px rgba(239, 68, 68, 0.25)',
      dotColor: '#ef4444',
    },
  };

  const item = configs[conf] || configs.low;
  const Icon = item.icon;

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        padding: '5px 12px',
        borderRadius: '999px',
        background: item.bg,
        border: `1px solid ${item.border}`,
        boxShadow: item.glow,
        backdropFilter: 'blur(8px)',
      }}
      title={item.subtext}
    >
      <div
        style={{
          width: '7px',
          height: '7px',
          borderRadius: '50%',
          backgroundColor: item.dotColor,
          boxShadow: `0 0 6px ${item.dotColor}`,
          animation: conf === 'high' ? 'pulseDot 2s infinite' : 'none',
        }}
      />
      <Icon size={14} color={item.color} />
      <span
        style={{
          fontSize: '0.74rem',
          fontWeight: 700,
          letterSpacing: '0.03em',
          color: item.color,
          textTransform: 'uppercase',
        }}
      >
        {item.label}
      </span>
    </div>
  );
}
