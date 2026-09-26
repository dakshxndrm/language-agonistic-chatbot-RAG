// AudioWorkletProcessor (NOT ScriptProcessorNode — CLAUDE.md §3 pin: the
// latter runs on the main thread and glitches under any UI work; this runs
// on the audio rendering thread). Its only job is a push-to-talk VU meter:
// posts an RMS level for each 128-sample render quantum so the UI can show
// the user their mic is live while the button is held. It does not touch
// the actual published audio path — LiveKit's own track publishing handles
// that — this is purely a local visual/gating signal.
class MicLevelProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) {
      let sumSquares = 0;
      for (let i = 0; i < channel.length; i++) sumSquares += channel[i] * channel[i];
      const rms = Math.sqrt(sumSquares / channel.length);
      this.port.postMessage({ rms });
    }
    return true;
  }
}

registerProcessor("mic-level-worklet", MicLevelProcessor);
