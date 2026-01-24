'use client'

import { useMemo } from 'react'
import { CheckCircle, ExternalLink, AlertCircle, Info } from 'lucide-react'

interface Citation {
  id: number
  url: string
  title: string
  snippet: string
}

interface Props {
  answer: string
  citations: Citation[]
  confidenceScores: Record<string, number>
}

function ConfidenceBadge({ label, score }: { label: string; score: number }) {
  let color: string
  let bg: string
  let Icon = CheckCircle

  if (score >= 0.8) {
    color = '#22c55e'
    bg = 'rgba(34,197,94,0.12)'
    Icon = CheckCircle
  } else if (score >= 0.5) {
    color = '#eab308'
    bg = 'rgba(234,179,8,0.12)'
    Icon = Info
  } else {
    color = '#ef4444'
    bg = 'rgba(239,68,68,0.12)'
    Icon = AlertCircle
  }

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
      style={{ background: bg, color }}
    >
      <Icon className="w-3 h-3" />
      {label}: {(score * 100).toFixed(0)}%
    </span>
  )
}

function renderAnswerWithCitations(text: string): React.ReactNode[] {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/)
    if (match) {
      return (
        <span key={i} className="citation-ref" title={`Citation ${match[1]}`}>
          {match[1]}
        </span>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export default function FinalAnswer({ answer, citations, confidenceScores }: Props) {
  const renderedAnswer = useMemo(() => renderAnswerWithCitations(answer), [answer])
  const hasConfidence = Object.keys(confidenceScores).length > 0
  const hasCitations = citations.length > 0

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Answer header */}
      <div className="flex items-center gap-2">
        <CheckCircle className="w-4 h-4 text-green-400" />
        <span className="text-sm font-semibold text-white">Research Complete</span>
      </div>

      {/* Confidence scores */}
      {hasConfidence && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(confidenceScores).map(([label, score]) => (
            <ConfidenceBadge key={label} label={label} score={score} />
          ))}
        </div>
      )}

      {/* Answer body */}
      <div
        className="rounded-xl p-5 border text-sm leading-7"
        style={{ background: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--foreground)' }}
      >
        <p>{renderedAnswer}</p>
      </div>

      {/* Citations */}
      {hasCitations && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
            Sources
          </h3>
          <div className="space-y-2">
            {citations.map(citation => (
              <div
                key={citation.id}
                className="flex gap-3 rounded-lg p-3 border"
                style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
              >
                <span
                  className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold mt-0.5"
                  style={{ background: 'rgba(59,130,246,0.15)', color: '#3b82f6' }}
                >
                  {citation.id}
                </span>
                <div className="flex-1 min-w-0 space-y-1">
                  <a
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs font-medium hover:underline truncate"
                    style={{ color: '#3b82f6' }}
                  >
                    {citation.title || citation.url}
                    <ExternalLink className="w-2.5 h-2.5 flex-shrink-0" />
                  </a>
                  {citation.snippet && (
                    <p className="text-xs line-clamp-2" style={{ color: 'var(--muted)' }}>
                      {citation.snippet}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
