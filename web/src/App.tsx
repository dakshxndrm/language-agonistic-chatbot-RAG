import { useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import "./index.css";
import VisualStage, { type VisualStageData } from "./VisualStage";

const API_URL = import.meta.env.VITE_API_URL ?? "";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  lang?: string;
  script?: string;
  citations?: string[];
  visual_stage?: VisualStageData;
  isStreaming?: boolean;
  timestamp: string;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedLang, setSelectedLang] = useState<string>("auto");
  const [visualData, setVisualData] = useState<VisualStageData | null>(null);
  const [showVisualPanel, setShowVisualPanel] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // LiveKit / Live Voice State
  const [room, setRoom] = useState<Room | null>(null);
  const [voiceStatus, setVoiceStatus] = useState<
    "disconnected" | "connecting" | "idle" | "listening" | "thinking" | "speaking"
  >("disconnected");
  const [isDictating, setIsDictating] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<any>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSubmitting]);

  // Clean up AudioContext & LiveKit room on unmount
  useEffect(() => {
    return () => {
      room?.disconnect();
      document.querySelectorAll("[data-livekit-audio]").forEach((el) => el.remove());
      audioCtxRef.current?.close().catch(() => {});
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, [room]);

  // Connect or disconnect LiveKit Voice Mode
  const handleToggleLiveVoice = async () => {
    if (voiceStatus !== "disconnected" && room) {
      // Disconnect
      room.disconnect();
      document.querySelectorAll("[data-livekit-audio]").forEach((el) => el.remove());
      setRoom(null);
      setVoiceStatus("disconnected");
      return;
    }

    try {
      setVoiceStatus("connecting");
      setError(null);

      // Unlock AudioContext if browser requires user gesture
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtx) {
          const ctx = new AudioCtx();
          if (ctx.state === "suspended") {
            await ctx.resume();
          }
        }
      } catch {
        // ignore audio context error
      }

      const resp = await fetch(`${API_URL}/livekit/token`, { method: "POST" });
      if (!resp.ok) throw new Error(`Token request failed: ${resp.status}`);
      const { token, url } = await resp.json();

      const r = new Room();

      // Track attached elements to prevent leaks
      const attachedEls = new Map<string, HTMLMediaElement>();

      const attachTrack = (track: Track) => {
        if (track.kind === Track.Kind.Audio) {
          const trackId = track.sid || `track-${Math.random().toString(36).substring(2, 9)}`;
          const existing = attachedEls.get(trackId);
          if (existing) {
            existing.remove();
          }
          const el = track.attach();
          el.setAttribute("data-livekit-audio", trackId);
          el.style.position = "fixed";
          el.style.top = "-9999px";
          el.style.left = "-9999px";
          el.style.width = "1px";
          el.style.height = "1px";
          el.style.opacity = "0.01";
          document.body.appendChild(el);
          attachedEls.set(trackId, el);
          el.play().catch((e) => console.warn("LiveKit audio play warning:", e));
        }
      };

      // Listen for data packets (visual stages or metadata) from Pipecat agent
      r.on(RoomEvent.DataReceived, (payload) => {
        try {
          const raw = JSON.parse(new TextDecoder().decode(payload));
          if (raw.event === "HISTORICAL_CONTEXT") {
            setVisualData(raw as VisualStageData);
          } else if (raw.event === "status") {
            if (raw.status === "thinking") {
              setVoiceStatus("thinking");
              const userQ = raw.question || raw.transcript;
              if (userQ) {
                setMessages((prev) => [
                  ...prev,
                  {
                    id: `voice-user-${Date.now()}`,
                    role: "user",
                    text: userQ,
                    timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                  },
                ]);
              }
            }
          } else if (raw.event === "answer" || raw.answer) {
            setMessages((prev) => [
              ...prev,
              {
                id: `msg-${Date.now()}`,
                role: "assistant",
                text: raw.answer,
                lang: raw.lang,
                script: raw.script,
                citations: raw.citations,
                visual_stage: raw.visual_stage,
                timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
              },
            ]);
            if (raw.visual_stage) {
              setVisualData(raw.visual_stage);
            }
            setVoiceStatus("speaking");
          } else if (raw.barge_in) {
            setVoiceStatus("listening");
          }
        } catch {
          // ignore non-JSON frames
        }
      });

      // Handle subscribed audio track from the assistant
      r.on(RoomEvent.TrackSubscribed, (track) => {
        attachTrack(track);
      });

      r.on(RoomEvent.TrackUnsubscribed, (track) => {
        const trackId = track.sid || "";
        const el = attachedEls.get(trackId);
        if (el) {
          el.remove();
          attachedEls.delete(trackId);
        }
        track.detach().forEach((e) => e.remove());
      });

      // Track active speakers for real-time visual feedback
      r.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const agentSpeaking = speakers.some(
          (s) => s.identity.startsWith("agent-") || s.name === "BHASHA Agent"
        );
        const userSpeaking = speakers.some((s) => s.isLocal);

        if (agentSpeaking) {
          setVoiceStatus("speaking");
        } else if (userSpeaking) {
          setVoiceStatus("listening");
        } else {
          setVoiceStatus((curr) => (curr === "speaking" ? "listening" : curr));
        }
      });

      r.on(RoomEvent.Disconnected, () => {
        attachedEls.forEach((el) => el.remove());
        attachedEls.clear();
        setVoiceStatus("disconnected");
        setRoom(null);
      });

      await r.connect(url, token);
      await r.localParticipant.setMicrophoneEnabled(true);

      // Start audio playback through LiveKit unlocker
      try {
        await r.startAudio();
      } catch (e) {
        console.warn("Could not auto-start audio:", e);
      }

      // Attach any remote tracks already published
      for (const p of r.remoteParticipants.values()) {
        for (const pub of p.trackPublications.values()) {
          if (pub.track) {
            attachTrack(pub.track);
          }
        }
      }

      setRoom(r);
      setVoiceStatus("listening");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setVoiceStatus("disconnected");
    }
  };

  // Text Submission with real-time SSE Streaming
  const handleSendText = async (textToSend?: string) => {
    const query = (textToSend ?? inputText).trim();
    if (!query || isSubmitting) return;

    setInputText("");
    setError(null);

    const userMessageId = `user-${Date.now()}`;
    const botMessageId = `bot-${Date.now() + 1}`;
    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // 1. Optimistically append user message & placeholder streaming assistant message
    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        role: "user",
        text: query,
        timestamp: timeStr,
      },
      {
        id: botMessageId,
        role: "assistant",
        text: "",
        isStreaming: true,
        timestamp: timeStr,
      },
    ]);

    setIsSubmitting(true);
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          lang: selectedLang !== "auto" ? selectedLang : undefined,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const rawData = trimmed.slice(6);
          try {
            const data = JSON.parse(rawData);

            if (data.event === "start") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botMessageId
                    ? { ...m, lang: data.lang, script: data.script }
                    : m
                )
              );
            } else if (data.event === "token") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botMessageId
                    ? { ...m, text: m.text + data.token }
                    : m
                )
              );
            } else if (data.event === "done") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botMessageId
                    ? {
                        ...m,
                        text: data.answer || m.text,
                        citations: data.citations,
                        visual_stage: data.visual_stage,
                        isStreaming: false,
                      }
                    : m
                )
              );
              if (data.visual_stage) {
                setVisualData(data.visual_stage);
              }
            } else if (data.event === "error") {
              throw new Error(data.error);
            }
          } catch {
            // ignore non-json line
          }
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) => (m.id === botMessageId ? { ...m, isStreaming: false } : m))
        );
      } else {
        // Fallback to standard /query if SSE is blocked
        try {
          const fallbackResp = await fetch(`${API_URL}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              question: query,
              lang: selectedLang !== "auto" ? selectedLang : undefined,
            }),
          });
          if (fallbackResp.ok) {
            const fallbackData = await fallbackResp.json();
            setMessages((prev) =>
              prev.map((m) =>
                m.id === botMessageId
                  ? {
                      ...m,
                      text: fallbackData.answer,
                      citations: fallbackData.citations,
                      visual_stage: fallbackData.visual_stage,
                      lang: fallbackData.lang,
                      script: fallbackData.script,
                      isStreaming: false,
                    }
                  : m
              )
            );
            if (fallbackData.visual_stage) {
              setVisualData(fallbackData.visual_stage);
            }
          } else {
            throw err;
          }
        } catch {
          setError(err instanceof Error ? err.message : String(err));
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botMessageId
                ? {
                    ...m,
                    text: "Sorry, I encountered an issue retrieving the answer. Please try again.",
                    isStreaming: false,
                  }
                : m
            )
          );
        }
      }
    } finally {
      setIsSubmitting(false);
      abortControllerRef.current = null;
    }
  };

  // Toggle Speech Dictation (Speech-to-Text) into the Input Box
  const handleToggleDictation = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError("Speech recognition is not natively supported in this browser. Please use Chrome/Edge or the Live Voice button.");
      return;
    }

    if (isDictating && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsDictating(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = selectedLang === "hi" ? "hi-IN" : "en-IN";

      recognition.onstart = () => {
        setIsDictating(true);
        setError(null);
      };

      recognition.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0].transcript)
          .join("");
        setInputText(transcript);
      };

      recognition.onerror = (e: any) => {
        setError(`Dictation error: ${e.error}`);
        setIsDictating(false);
      };

      recognition.onend = () => {
        setIsDictating(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (e: any) {
      setError(`Could not start dictation: ${e.message}`);
      setIsDictating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  };

  const starterQuestions = [
    { label: "👑 Reign of Emperor Ashoka", q: "Who was Ashoka and what was his role in the Maurya Empire?" },
    { label: "⚔️ The Kalinga War", q: "Which war caused Ashoka to renounce violence and adopt Buddhism?" },
    { label: "📜 अशोक का धम्म (Hindi)", q: "अशोक का धम्म क्या था और उन्होंने इसका प्रचार कैसे किया?" },
    { label: "🏛️ Chandragupta Maurya & Chanakya", q: "Who was the grandfather of Ashoka and what empire did he found?" },
  ];

  return (
    <>
      {/* Top App Header */}
      <header className="app-header">
        <div className="brand-container">
          <span className="brand-logo">🕉️</span>
          <div>
            <h1 className="brand-title">BHASHA</h1>
            <div className="brand-subtitle">Multimodal Bilingual Speech & Knowledge Assistant</div>
          </div>
        </div>

        <div className="header-controls">
          {/* Language Mode Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "14px" }}>🌐</span>
            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              style={{
                padding: "4px 8px",
                borderRadius: "8px",
                border: "1px solid var(--border)",
                background: "var(--bg)",
                color: "var(--text-h)",
                fontSize: "12px",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              <option value="auto">Auto Language</option>
              <option value="en">English (Pure)</option>
              <option value="hi">Hindi (हिन्दी)</option>
              <option value="hi-en">Hinglish</option>
            </select>
          </div>

          {/* Toggle Visual Stage Sidebar */}
          <button
            onClick={() => setShowVisualPanel((prev) => !prev)}
            className="chip-btn"
            style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}
          >
            <span>🎭</span>
            <span>{showVisualPanel ? "Hide Map & Graph" : "Show Map & Graph"}</span>
          </button>
        </div>
      </header>

      {/* Main Split View Area */}
      <div className="main-viewport">
        {/* Left/Center: Chat Timeline */}
        <section className="chat-section">
          <div className="chat-timeline">
            {error && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "rgba(239, 68, 68, 0.1)",
                  border: "1px solid rgba(239, 68, 68, 0.3)",
                  borderRadius: "10px",
                  color: "#ef4444",
                  fontSize: "13px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span>⚠️ {error}</span>
                <button
                  onClick={() => setError(null)}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "#ef4444", fontWeight: 700 }}
                >
                  ✕
                </button>
              </div>
            )}

            {messages.length === 0 ? (
              <div className="welcome-container">
                <div style={{ fontSize: "42px", marginBottom: "8px" }}>🕉️</div>
                <h2 className="welcome-title">Ask about Indian History</h2>
                <p className="welcome-desc">
                  Voice-first, bilingual assistant powered by Knowledge Graphs and Verified Sources.
                  Speak or type below to explore historical events, dynasties, and artifacts.
                </p>

                <div className="prompt-suggestions">
                  {starterQuestions.map((item, idx) => (
                    <div
                      key={idx}
                      className="suggestion-card"
                      onClick={() => handleSendText(item.q)}
                    >
                      <span style={{ fontSize: "16px" }}>{item.label.split(" ")[0]}</span>
                      <div>
                        <div style={{ fontWeight: 600 }}>{item.label.split(" ").slice(1).join(" ")}</div>
                        <div style={{ fontSize: "11px", color: "var(--text)", marginTop: "2px" }}>
                          {item.q.slice(0, 48)}...
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) =>
                msg.role === "user" ? (
                  <div key={msg.id} className="message-row message-row-user">
                    <div className="user-bubble">
                      <div>{msg.text}</div>
                      <div style={{ fontSize: "10px", opacity: 0.75, textAlign: "right", marginTop: "4px" }}>
                        {msg.timestamp}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div key={msg.id} className="message-row message-row-assistant">
                    <div className="bot-avatar">🕉️</div>
                    <div className="bot-bubble">
                      <div className="bot-text">
                        {msg.text}
                        {msg.isStreaming && <span className="streaming-cursor" />}
                      </div>

                      {/* Metadata Chips: Language, Citations, Graph inspector */}
                      {(!msg.isStreaming || msg.citations) && (
                        <div className="bot-meta">
                          {msg.lang && (
                            <span className="chip chip-lang">
                              🌐 {msg.lang.toUpperCase()} ({msg.script || "Native"})
                            </span>
                          )}

                          {msg.citations && msg.citations.length > 0 && (
                            <>
                              <span style={{ fontSize: "11px", color: "var(--text)" }}>Sources:</span>
                              {msg.citations.map((cite, i) => (
                                <span key={i} className="chip chip-citation">
                                  📚 {cite}
                                </span>
                              ))}
                            </>
                          )}

                          {msg.visual_stage && (
                            <button
                              onClick={() => {
                                setVisualData(msg.visual_stage!);
                                setShowVisualPanel(true);
                              }}
                              className="chip-btn"
                              style={{ marginLeft: "auto", fontSize: "11px" }}
                            >
                              🎭 Open Graph & Map
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )
              )
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Gemini-Style Floating Bottom Dock */}
          <div className="gemini-dock-wrapper">
            <div className="gemini-dock">
              {/* Mic Dictation Button (Speech-to-Text into Input) */}
              <button
                type="button"
                onClick={handleToggleDictation}
                className={`dock-btn ${isDictating ? "dock-btn-mic-active" : ""}`}
                title={isDictating ? "Stop voice dictation" : "Dictate with microphone"}
              >
                {isDictating ? "⏹️" : "🎙️"}
              </button>

              {/* Text Input Box */}
              <input
                type="text"
                className="dock-input"
                placeholder={
                  isDictating
                    ? "Listening to speech..."
                    : "Ask about Indian history in English, Hindi, or Hinglish..."
                }
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isSubmitting}
              />

              {/* Dedicated Live Voice Mode Pill Button */}
              <button
                type="button"
                onClick={handleToggleLiveVoice}
                className={`live-voice-pill ${
                  voiceStatus === "listening"
                    ? "active-listening"
                    : voiceStatus === "thinking"
                    ? "active-processing"
                    : voiceStatus === "speaking"
                    ? "active-speaking"
                    : ""
                }`}
                title="Bidirectional live voice conversation"
              >
                <span>{voiceStatus === "disconnected" ? "⚡" : "🔊"}</span>
                <span>
                  {voiceStatus === "disconnected"
                    ? "Live Voice"
                    : voiceStatus === "connecting"
                    ? "Connecting..."
                    : voiceStatus === "listening"
                    ? "Listening..."
                    : voiceStatus === "thinking"
                    ? "Thinking..."
                    : voiceStatus === "speaking"
                    ? "Speaking..."
                    : "Live Voice"}
                </span>
              </button>

              {/* Send / Stop Button */}
              {isSubmitting ? (
                <button
                  type="button"
                  onClick={() => abortControllerRef.current?.abort()}
                  className="dock-btn"
                  title="Stop generating"
                >
                  ⏹️
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => handleSendText()}
                  disabled={!inputText.trim()}
                  className="dock-btn dock-btn-send"
                  title="Send message"
                >
                  ➤
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Right Side: Multimodal Visual Stage Panel */}
        {showVisualPanel && (
          <aside className="visual-stage-sidebar">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontWeight: 700, fontSize: "14px", color: "var(--text-h)" }}>
                🎭 Multimodal Visual Stage
              </div>
              <button
                onClick={() => setShowVisualPanel(false)}
                className="chip-btn"
                style={{ fontSize: "11px", padding: "2px 6px" }}
              >
                ✕ Close
              </button>
            </div>

            <VisualStage
              data={visualData}
              isSpeaking={voiceStatus === "speaking"}
              citationsList={
                visualData?.citation ? [visualData.citation.chunk_id] : []
              }
            />
          </aside>
        )}
      </div>
    </>
  );
}
