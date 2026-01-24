'use client'

import { useState, useRef, useCallback } from 'react'
import { Send, Loader2, Brain } from 'lucide-react'
import * as ScrollArea from '@radix-ui/react-scroll-area'
import AgentThoughtChain from './AgentThoughtChain'
import FinalAnswer from './FinalAnswer'
import SessionSidebar from './SessionSidebar'
import { startResearch, createEventSource, type StreamEvent } from '@/lib/api'

export default function ResearchInterface() {
  const [question, setQuestion] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [agentEvents, setAgentEvents] = useState<StreamEvent[]>([])
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null)
  const [citations, setCitations] = useState<Array<{ id: number; url: string; title: string; snippet: string }>>([])
  const [confidenceScores, setConfidenceScores] = useState<Record<string, number>>({})
  const [error, setError] = useState<string | null>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

  const handleSubmit = useCallback(async () => {
    if (!question.trim() || isLoading) return

    setIsLoading(true)
    setAgentEvents([])
    setFinalAnswer(null)
    setCitations([])
    setConfidenceScores({})
    setError(null)

    try {
      const { session_id, stream_url } = await startResearch(question.trim())
      setSessionId(session_id)

      if (cleanupRef.current) {
        cleanupRef.current()
      }

      cleanupRef.current = createEventSource(stream_url, (event: StreamEvent) => {
        if (event.type === 'agent_update') {
          setAgentEvents(prev => [...prev, event])
        } else if (event.type === 'final') {
          const data = event.data as {
            answer?: string
            citations?: Array<{ id: number; url: string; title: string; snippet: string }>
            confidence_scores?: Record<string, number>
          }
          setFinalAnswer(data.answer ?? event.content)
          setCitations(data.citations ?? [])
          setConfidenceScores(data.confidence_scores ?? {})
          setIsLoading(false)
        } else if (event.type === 'error') {
          setError(event.content)
          setIsLoading(false)
        }
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start research')
      setIsLoading(false)
    }
  }, [question, isLoading])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSessionLoad = (loadedQuestion: string) => {
    setQuestion(loadedQuestion)
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--background)' }}>
      {/* Sidebar */}
      <SessionSidebar onSessionSelect={handleSessionLoad} activeSessionId={sessionId} />

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel: input + answer */}
        <div className="flex flex-col flex-1 overflow-hidden border-r" style={{ borderColor: 'var(--border)' }}>
          {/* Header */}
          <div className="flex items-center gap-3 px-6 py-4 border-b" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
            <Brain className="w-5 h-5 text-blue-400" />
            <span className="font-semibold text-sm tracking-wide text-white">Meridian</span>
            <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(59,130,246,0.15)', color: '#3b82f6' }}>
              Research Assistant
            </span>
          </div>

          {/* Answer area */}
          <ScrollArea.Root className="flex-1 overflow-hidden">
            <ScrollArea.Viewport className="h-full w-full">
              <div className="p-6 space-y-6">
                {!finalAnswer && !isLoading && agentEvents.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-64 text-center space-y-3">
                    <div className="w-12 h-12 rounded-full flex items-center justify-center" style={{ background: 'rgba(59,130,246,0.1)' }}>
                      <Brain className="w-6 h-6 text-blue-400" />
                    </div>
                    <p className="text-sm" style={{ color: 'var(--muted)' }}>
                      Ask a question to start a deep research session
                    </p>
                    <p className="text-xs" style={{ color: 'var(--muted)' }}>
                      Meridian will deploy multiple AI agents to research and synthesize an answer
                    </p>
                  </div>
                )}

                {isLoading && agentEvents.length === 0 && (
                  <div className="flex items-center gap-3 p-4 rounded-lg" style={{ background: 'var(--surface)' }}>
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                    <span className="text-sm" style={{ color: 'var(--muted)' }}>Initializing research agents...</span>
                  </div>
                )}

                {error && (
                  <div className="p-4 rounded-lg border text-sm" style={{ background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.3)', color: '#f87171' }}>
                    <strong>Error:</strong> {error}
                  </div>
                )}

                {finalAnswer && (
                  <FinalAnswer
                    answer={finalAnswer}
                    citations={citations}
                    confidenceScores={confidenceScores}
                  />
                )}
              </div>
            </ScrollArea.Viewport>
            <ScrollArea.Scrollbar orientation="vertical" className="flex w-1.5 touch-none select-none p-0.5">
              <ScrollArea.Thumb className="relative flex-1 rounded-full" style={{ background: 'var(--border)' }} />
            </ScrollArea.Scrollbar>
          </ScrollArea.Root>

          {/* Input area */}
          <div className="p-4 border-t" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
            <div className="relative rounded-xl border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--surface-elevated)' }}>
              <textarea
                value={question}
                onChange={e => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a research question... (Cmd+Enter to submit)"
                rows={3}
                disabled={isLoading}
                className="w-full resize-none bg-transparent px-4 pt-3 pb-10 text-sm outline-none placeholder:text-[var(--muted)] text-[var(--foreground)] disabled:opacity-50"
              />
              <div className="absolute bottom-3 right-3 flex items-center gap-2">
                <span className="text-xs" style={{ color: 'var(--muted)' }}>
                  {question.length > 0 ? `${question.length} chars` : 'Cmd+Enter'}
                </span>
                <button
                  onClick={handleSubmit}
                  disabled={!question.trim() || isLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{ background: '#3b82f6', color: 'white' }}
                >
                  {isLoading ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Send className="w-3.5 h-3.5" />
                  )}
                  {isLoading ? 'Researching...' : 'Research'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right panel: agent thought chain */}
        <div className="w-96 flex flex-col overflow-hidden" style={{ background: 'var(--surface)' }}>
          <div className="flex items-center gap-2 px-4 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
            <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`} />
            <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
              Agent Thought Chain
            </span>
            {agentEvents.length > 0 && (
              <span className="ml-auto text-xs px-1.5 py-0.5 rounded-full" style={{ background: 'var(--border)', color: 'var(--muted)' }}>
                {agentEvents.length}
              </span>
            )}
          </div>
          <AgentThoughtChain events={agentEvents} isLoading={isLoading} />
        </div>
      </div>
    </div>
  )
}
