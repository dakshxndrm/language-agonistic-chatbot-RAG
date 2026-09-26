import { useCallback, useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import "./index.css";

// Same-origin by default; override for local dev against a backend on a
// different port (`VITE_API_URL=http://localhost:8000`).
const API_URL = import.meta.env.VITE_API_URL ?? "";

interface TurnData {
  transcript: string;
  lang: string;
  script: string;
  answer: string;
  citations: string[];
  fidelity_passed: boolean;
}

export default function App() {
  const [room, setRoom] = useState<Room | null>(null);
  const [connected, setConnected] = useState(false);
  const [talking, setTalking] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [turn, setTurn] = useState<TurnData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const workletCleanupRef = useRef<(() => void) | null>(null);

  const connect = useCallback(async () => {
    try {
      const resp = await fetch(`${API_URL}/livekit/token`, { method: "POST" });
      if (!resp.ok) throw new Error(`token request failed: ${resp.status}`);
      const { token, url } = await resp.json();

      const r = new Room();
      r.on(RoomEvent.DataReceived, (payload) => {
        try {
          setTurn(JSON.parse(new TextDecoder().decode(payload)));
        } catch {
          // Non-JSON data messages aren't ours to render — ignore rather
          // than crash the UI on an unrelated message.
        }
      });
      r.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio && audioRef.current) {
          track.attach(audioRef.current);
        }
      });
      r.on(RoomEvent.Disconnected, () => setConnected(false));

      await r.connect(url, token);
      // Published muted — push-to-talk unmutes it, per CLAUDE.md's UX pin.
      await r.localParticipant.setMicrophoneEnabled(false);

      setRoom(r);
      setConnected(true);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    return () => {
      room?.disconnect();
      workletCleanupRef.current?.();
    };
  }, [room]);

  const startTalking = useCallback(async () => {
    if (!room) return;
    await room.localParticipant.setMicrophoneEnabled(true);
    setTalking(true);

    // Mic-level VU meter via AudioWorklet (NOT ScriptProcessorNode — §3
    // pin). Purely a visual aid; it does not touch the published track.
    const micPub = room.localParticipant.getTrackPublication(Track.Source.Microphone);
    const mediaTrack = micPub?.track?.mediaStreamTrack;
    if (mediaTrack) {
      const ctx = new AudioContext();
      await ctx.audioWorklet.addModule("/mic-level-worklet.js");
      const source = ctx.createMediaStreamSource(new MediaStream([mediaTrack]));
      const node = new AudioWorkletNode(ctx, "mic-level-worklet");
      node.port.onmessage = (e) => setMicLevel(e.data.rms);
      source.connect(node);
      workletCleanupRef.current = () => {
        node.port.onmessage = null;
        ctx.close();
      };
    }
  }, [room]);

  const stopTalking = useCallback(async () => {
    if (!room) return;
    await room.localParticipant.setMicrophoneEnabled(false);
    setTalking(false);
    setMicLevel(0);
    workletCleanupRef.current?.();
    workletCleanupRef.current = null;
  }, [room]);

  return (
    <main style={{ maxWidth: 640, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>BHASHA</h1>
      <p>Ask about Indian history — in English, Hindi, or Hinglish.</p>

      {!connected ? (
        <button onClick={connect}>Connect</button>
      ) : (
        <button
          onMouseDown={startTalking}
          onMouseUp={stopTalking}
          onTouchStart={startTalking}
          onTouchEnd={stopTalking}
          style={{
            padding: "1rem 2rem",
            background: talking ? "#c0392b" : "#2c3e50",
            color: "white",
            border: "none",
            borderRadius: 8,
          }}
        >
          {talking ? `Listening… (${micLevel.toFixed(2)})` : "Hold to talk"}
        </button>
      )}

      {error && <p style={{ color: "red" }}>{error}</p>}

      {turn && (
        <section style={{ marginTop: "2rem", lineHeight: 1.6 }}>
          <p>
            <strong>You said</strong> ({turn.lang}/{turn.script}): {turn.transcript}
          </p>
          <p>
            <strong>Answer</strong> {!turn.fidelity_passed && "⚠ script check failed"}:{" "}
            {turn.answer}
          </p>
          {turn.citations.length > 0 && (
            <p>
              <strong>Sources:</strong> {turn.citations.join(", ")}
            </p>
          )}
        </section>
      )}

      <audio ref={audioRef} autoPlay />
    </main>
  );
}
