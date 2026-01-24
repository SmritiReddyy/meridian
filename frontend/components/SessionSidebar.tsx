'use client'

import { useEffect, useState, useCallback } from 'react'
import { Clock, MessageSquare, ChevronRight, RefreshCw } from 'lucide-react'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import { getSessions, type SessionSummary } from '@/lib/api'

interface Props {
  onSessionSelect: (question: string) => void
  activeSessionId: string | null
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays === 1) return 'Yesterday'
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  } catch {
    return dateStr
  }
}

export default function SessionSidebar({ onSessionSelect, activeSessionId }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadSessions = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await getSessions()
      setSessions(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  return (
    <div
      className="w-64 flex flex-col border-r"
      style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <Clock className="w-4 h-4" style={{ color: 'var(--muted)' }} />
        <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
          Sessions
        </span>
        <button
          onClick={loadSessions}
          disabled={isLoading}
          className="ml-auto p-1 rounded transition-colors hover:bg-white/5 disabled:opacity-40"
          title="Refresh sessions"
        >
          <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} style={{ color: 'var(--muted)' }} />
        </button>
      </div>

      {/* Session list */}
      <ScrollArea.Root className="flex-1 overflow-hidden">
        <ScrollArea.Viewport className="h-full w-full">
          <div className="p-2 space-y-1">
            {error && (
              <p className="text-xs px-2 py-1.5 rounded" style={{ color: '#f87171', background: 'rgba(239,68,68,0.08)' }}>
                {error}
              </p>
            )}

            {!isLoading && !error && sessions.length === 0 && (
              <div className="flex flex-col items-center justify-center py-10 text-center space-y-2">
                <MessageSquare className="w-6 h-6" style={{ color: 'var(--muted)' }} />
                <p className="text-xs" style={{ color: 'var(--muted)' }}>
                  No sessions yet
                </p>
              </div>
            )}

            {sessions.map(session => (
              <button
                key={session.session_id}
                onClick={() => onSessionSelect(session.question)}
                className="w-full text-left rounded-lg px-3 py-2.5 transition-colors group"
                style={{
                  background: activeSessionId === session.session_id
                    ? 'rgba(59,130,246,0.12)'
                    : 'transparent',
                  border: activeSessionId === session.session_id
                    ? '1px solid rgba(59,130,246,0.25)'
                    : '1px solid transparent',
                }}
              >
                <div className="flex items-start gap-2">
                  <div className="flex-1 min-w-0 space-y-1">
                    <p
                      className="text-xs font-medium truncate leading-snug"
                      style={{ color: activeSessionId === session.session_id ? '#60a5fa' : 'var(--foreground)' }}
                    >
                      {session.question}
                    </p>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs" style={{ color: 'var(--muted)' }}>
                        {formatDate(session.created_at)}
                      </span>
                      {session.status && (
                        <span
                          className="text-xs px-1.5 py-0.5 rounded-full"
                          style={{
                            background: session.status === 'completed'
                              ? 'rgba(34,197,94,0.12)'
                              : 'rgba(234,179,8,0.12)',
                            color: session.status === 'completed' ? '#22c55e' : '#eab308',
                          }}
                        >
                          {session.status}
                        </span>
                      )}
                    </div>
                  </div>
                  <ChevronRight
                    className="w-3 h-3 flex-shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: 'var(--muted)' }}
                  />
                </div>
              </button>
            ))}
          </div>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar orientation="vertical" className="flex w-1.5 touch-none select-none p-0.5">
          <ScrollArea.Thumb className="relative flex-1 rounded-full" style={{ background: 'var(--border)' }} />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
    </div>
  )
}
