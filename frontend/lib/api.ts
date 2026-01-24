export interface StreamEvent {
  type: string
  agent: string
  content: string
  data: Record<string, unknown>
  timestamp: string
}

export interface SessionSummary {
  session_id: string
  question: string
  created_at: string
  status?: string
}

export interface ResearchResponse {
  session_id: string
  stream_url: string
}

const API_BASE = '/api'

export async function startResearch(question: string): Promise<ResearchResponse> {
  const res = await fetch(`${API_BASE}/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(`Research request failed (${res.status}): ${text}`)
  }

  return res.json()
}

export function createEventSource(
  url: string,
  onEvent: (event: StreamEvent) => void
): () => void {
  // Use fetch + ReadableStream for SSE to work with Next.js rewrites
  const controller = new AbortController()

  const connect = async () => {
    try {
      const res = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: 'text/event-stream', 'Cache-Control': 'no-cache' },
      })

      if (!res.ok || !res.body) {
        onEvent({
          type: 'error',
          agent: 'system',
          content: `Stream connection failed (${res.status})`,
          data: {},
          timestamp: new Date().toISOString(),
        })
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        let eventType = ''
        let dataLines: string[] = []

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim())
          } else if (line === '') {
            if (dataLines.length > 0) {
              const rawData = dataLines.join('\n')
              try {
                const parsed = JSON.parse(rawData) as Partial<StreamEvent>
                const event: StreamEvent = {
                  type: parsed.type ?? eventType ?? 'unknown',
                  agent: parsed.agent ?? '',
                  content: parsed.content ?? '',
                  data: (parsed.data as Record<string, unknown>) ?? parsed ?? {},
                  timestamp: parsed.timestamp ?? new Date().toISOString(),
                }
                onEvent(event)
              } catch {
                // Non-JSON data line — emit as raw content
                onEvent({
                  type: eventType || 'raw',
                  agent: '',
                  content: rawData,
                  data: {},
                  timestamp: new Date().toISOString(),
                })
              }
              eventType = ''
              dataLines = []
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        onEvent({
          type: 'error',
          agent: 'system',
          content: err instanceof Error ? err.message : 'Stream error',
          data: {},
          timestamp: new Date().toISOString(),
        })
      }
    }
  }

  connect()

  return () => {
    controller.abort()
  }
}

export async function getSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_BASE}/sessions`, {
    headers: { 'Content-Type': 'application/json' },
  })

  if (!res.ok) {
    throw new Error(`Failed to fetch sessions (${res.status})`)
  }

  const data = await res.json()
  // Handle both {sessions: [...]} and plain array responses
  if (Array.isArray(data)) return data
  if (Array.isArray(data.sessions)) return data.sessions
  return []
}
