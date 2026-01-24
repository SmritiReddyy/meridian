'use client'

import { useEffect, useRef } from 'react'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import { Bot, Search, FlaskConical, Layers, AlertTriangle } from 'lucide-react'
import type { StreamEvent } from '@/lib/api'

const AGENT_CONFIG: Record<string, { label: string; color: string; bg: string; icon: React.ReactNode }> = {
  planner: {
    label: 'Planner',
    color: '#3b82f6',
    bg: 'rgba(59,130,246,0.12)',
    icon: <Layers className="w-3 h-3" />,
  },
  researcher: {
    label: 'Researcher',
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.12)',
    icon: <Search className="w-3 h-3" />,
  },
  critic: {
    label: 'Critic',
    color: '#eab308',
    bg: 'rgba(234,179,8,0.12)',
    icon: <AlertTriangle className="w-3 h-3" />,
  },
  synthesizer: {
    label: 'Synthesizer',
    color: '#a855f7',
    bg: 'rgba(168,85,247,0.12)',
    icon: <FlaskConical className="w-3 h-3" />,
  },
}

const DEFAULT_AGENT = {
  label: 'Agent',
  color: '#6b7280',
  bg: 'rgba(107,114,128,0.12)',
  icon: <Bot className="w-3 h-3" />,
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return ts
  }
}

interface Props {
  events: StreamEvent[]
  isLoading: boolean
}

export default function AgentThoughtChain({ events, isLoading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  return (
    <ScrollArea.Root className="flex-1 overflow-hidden">
      <ScrollArea.Viewport className="h-full w-full">
        <div className="p-3 space-y-2">
          {events.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center py-16 text-center space-y-2">
              <Bot className="w-8 h-8" style={{ color: 'var(--muted)' }} />
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                Agent activity will appear here
              </p>
            </div>
          )}

          {events.map((event, idx) => {
            const agentKey = (event.agent ?? '').toLowerCase()
            const cfg = AGENT_CONFIG[agentKey] ?? DEFAULT_AGENT

            return (
              <div
                key={idx}
                className="rounded-lg p-3 space-y-2 animate-slide-up"
                style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}
              >
                {/* Header row */}
                <div className="flex items-center gap-2">
                  <span
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
                    style={{ background: cfg.bg, color: cfg.color }}
                  >
                    {cfg.icon}
                    {cfg.label}
                  </span>
                  <span className="ml-auto text-xs tabular-nums" style={{ color: 'var(--muted)' }}>
                    {formatTimestamp(event.timestamp)}
                  </span>
                </div>

                {/* Content */}
                <p className="text-xs leading-relaxed" style={{ color: 'var(--foreground)', opacity: 0.85 }}>
                  {event.content}
                </p>
              </div>
            )
          })}

          {isLoading && (
            <div
              className="rounded-lg p-3 animate-pulse"
              style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="h-4 w-16 rounded-full" style={{ background: 'var(--border)' }} />
                <div className="ml-auto h-3 w-12 rounded" style={{ background: 'var(--border)' }} />
              </div>
              <div className="space-y-1.5">
                <div className="h-2.5 rounded w-full" style={{ background: 'var(--border)' }} />
                <div className="h-2.5 rounded w-4/5" style={{ background: 'var(--border)' }} />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar orientation="vertical" className="flex w-1.5 touch-none select-none p-0.5">
        <ScrollArea.Thumb className="relative flex-1 rounded-full" style={{ background: 'var(--border)' }} />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  )
}
