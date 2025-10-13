#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Feedback suppression module for InterCom, built *over the buffering layer*.

It implements two simple strategies to mitigate acoustic feedback/echo when
using speakers + mic (no headphones):

1) Delay-line subtraction ("delay"): subtract a delayed/attenuated copy of the
   previously played signal from the newly recorded mic signal.

2) Adaptive FIR with LMS ("lms"): adapt filter coefficients that model the
   loudspeaker→room→mic acoustic path, and subtract the predicted echo.

This module **inherits from Buffering** so it keeps the jitter-hiding buffer
and adds suppression right where it matters: cleaning the *recorded* signal
(ADC) before sending it to the interlocutor.

Parameters are exposed via argparse flags added to the shared `minimal.parser`.

Usage examples (run on each peer, replacing IPs as needed):

  # Pure buffer (reference)
  python buffering.py --show_stats

  # Buffer + delay-line echo suppression (50 ms, a=0.8)
  python feedback_suppression.py --fs_method delay --fs_delay_ms 50 --fs_attenuation 0.8 --show_stats

  # Buffer + LMS adaptive filter (64 taps, mu=0.001)
  python feedback_suppression.py --fs_method lms --fs_lms_len 64 --fs_lms_mu 0.001 --show_stats

Notes:
- Delay in ms is translated to an integer number of chunks using chunk_time.
- LMS runs per‑channel. Keep `--fs_lms_len` modest (e.g., 32–128) for real‑time.
- Signals are kept/clipped in int16 range after processing.
"""

import collections
import logging
import numpy as np
import minimal
import buffer as buffering_mod  # your Buffering class lives here

# --------------------------- CLI flags (argparse) --------------------------- #
minimal.parser.add_argument("--fs_method", type=str, default="off",
                            choices=["off", "delay", "lms"],
                            help="Feedback suppression strategy")
minimal.parser.add_argument("--fs_delay_ms", type=float, default=50.0,
                            help="Echo path delay in milliseconds (used by 'delay' and as initial lag ref for 'lms')")
minimal.parser.add_argument("--fs_attenuation", type=float, default=0.8,
                            help="Attenuation factor 'a' for the delayed echo estimate (0..1)")
minimal.parser.add_argument("--fs_lms_len", type=int, default=64,
                            help="LMS adaptive FIR length (number of taps)")
minimal.parser.add_argument("--fs_lms_mu", type=float, default=1e-3,
                            help="LMS learning rate (mu)")


# ------------------------------ Core class --------------------------------- #
class FeedbackBuffer(buffering_mod.Buffering):
    """Buffering + feedback suppression.

    Strategy:
    - Maintain a history of recently played chunks (what goes to DAC).
    - When we capture a new microphone chunk (ADC), estimate the portion that
      actually comes from our own loudspeakers (echo), and subtract it.
    - Then we send the cleaned chunk to the peer.

    We override only the parts where ADC is processed before pack/send and
    where played chunks should be registered into the history.
    """

    def __init__(self):
        super().__init__()
        self.method = minimal.args.fs_method
        self.atten = float(min(max(minimal.args.fs_attenuation, 0.0), 1.0))
        self.lms_len = int(max(1, minimal.args.fs_lms_len))
        self.mu = float(minimal.args.fs_lms_mu)

        # Convert delay in ms to a number of *chunks* (integer)
        delay_chunks = max(0, int(round((minimal.args.fs_delay_ms / 1000.0) / self.chunk_time)))
        self.delay_chunks = delay_chunks

        # History of played chunks for echo reference
        self.play_history = collections.deque(maxlen=max(self.delay_chunks + self.lms_len + 2, 4))
        # Pre-fill with zeros to avoid conditionals
        zero = self.generate_zero_chunk()
        for _ in range(self.play_history.maxlen):
            self.play_history.append(zero.copy())

        # LMS state per channel (w shape: [channels, taps])
        self._init_lms_state()

        logging.info(f"[FS] method={self.method} delay_chunks={self.delay_chunks} a={self.atten} lms_len={self.lms_len} mu={self.mu}")

    # ---------------------------- LMS helpers ---------------------------- #
    def _init_lms_state(self):
        ch = minimal.args.number_of_channels
        self.w = np.zeros((ch, self.lms_len), dtype=np.float64)
        # We will assemble the reference vector x[n] from the played history.

    def _get_reference_block(self):
        """Build the reference block x[n] for echo path (shape: [frames, channels]).
        Uses the *delayed* last played chunk as starting point and may extend
        with previous chunks to provide lms_len samples per channel.
        """
        frames = minimal.args.frames_per_chunk
        ch = minimal.args.number_of_channels

        # We want at least lms_len samples; concatenate delayed chunk + previous
        # chunks until we reach frames worth of samples (per chunk we filter
        # frame-by-frame using a sliding window). For simplicity: take the
        # delayed chunk only; LMS will still work but with reduced context.
        # To improve quality, we build a length-frames vector with the delayed
        # chunk; for LMS taps > 1, we will index into past samples within that
        # chunk (no cross-chunk concat to keep real-time cost low).

        # Get the delayed chunk from history (int16), shape [frames, ch]
        delayed_idx = -(self.delay_chunks + 1)
        ref_chunk = self.play_history[delayed_idx]
        if ref_chunk.dtype != np.int16:
            ref_chunk = ref_chunk.astype(np.int16, copy=False)
        return ref_chunk

    # ------------------------ Suppression methods ------------------------ #
    def _suppress_delay(self, adc_chunk):
        """Simple delay-line subtraction: adc - a * delayed_played."""
        ref = self._get_reference_block().astype(np.int32)
        adc = adc_chunk.astype(np.int32)
        est = (self.atten * ref).astype(np.int32)
        clean = adc - est
        # clip back to int16
        np.clip(clean, -32768, 32767, out=clean)
        return clean.astype(np.int16)

    def _suppress_lms(self, adc_chunk):
        """Block LMS per-chunk, per-channel (simple implementation).

        y[n] = sum_k w_k * x[n-k] ; e[n] = d[n] - y[n] ; w <- w + mu * e[n] * x[n]
        Where d[n] ~ adc (mic), x[n] ~ reference from played chunk (delayed).
        """
        d = adc_chunk.astype(np.float64)  # desired (mic)
        x_chunk = self._get_reference_block().astype(np.float64)
        frames, ch = d.shape[0], d.shape[1]

        # Output arrays
        y = np.zeros_like(d)
        e = np.zeros_like(d)

        # Simple per-sample LMS using only intra-chunk context for speed
        # (if lms_len > frames, we'll effectively use up to `frames`).
        taps = min(self.lms_len, frames)
        for c in range(ch):
            # Zero-padded buffer of the reference for channel c
            xc = x_chunk[:, c]
            wc = self.w[c, :taps]
            # Iterate samples (tight loop; taps small keeps cost bounded)
            # y[n] = w^T x_n, where x_n = [x[n], x[n-1], ..., x[n-taps+1]]
            for n in range(frames):
                start = max(0, n - taps + 1)
                x_vec = xc[start:n+1][::-1]  # length <= taps
                # If shorter than taps, pad (implicitly aligns with wc[:len])
                y_val = float(np.dot(wc[:x_vec.size], x_vec))
                y[n, c] = y_val
                e_val = d[n, c] - y_val
                e[n, c] = e_val
                # LMS update: w <- w + mu * e * x
                wc[:x_vec.size] += self.mu * e_val * x_vec
            # write back updated coeffs (only first `taps` meaningful)
            self.w[c, :taps] = wc

        clean = d - y
        # Clip back to int16
        np.clip(clean, -32768, 32767, out=clean)
        return clean.astype(np.int16)

    def suppress_feedback(self, adc_chunk):
        if self.method == "off":
            return adc_chunk
        if self.method == "delay":
            return self._suppress_delay(adc_chunk)
        if self.method == "lms":
            return self._suppress_lms(adc_chunk)
        return adc_chunk

    # --------------------------- Overridden I/O --------------------------- #
    def play_chunk(self, DAC, chunk):
        """Override to also push played audio into history for echo reference."""
        super().play_chunk(DAC, chunk)
        # The super reshapes and writes to DAC. We must append the *actually*
        # played chunk to the history for future echo estimation.
        # DAC is a numpy view; take a copy to freeze samples in history.
        self.play_history.append(DAC.copy())

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        """Called by PortAudio for mic input + speaker output (full-duplex)."""
        # 1) Clean the recorded chunk before sending
        clean_ADC = self.suppress_feedback(ADC)

        # 2) Send cleaned mic data (as in Buffering)
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS
        packed_chunk = self.pack(self.chunk_number, clean_ADC)
        self.send(packed_chunk)

        # 3) Play next chunk from buffer (remote audio)
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)

    def _read_IO_and_play(self, DAC, frames, time, status):
        """File-as-mic path (useful for repeatable tests)."""
        # Read local reference audio (acts as our mic speech)
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS
        read_chunk = self.read_chunk_from_file()

        # Clean it as if it were mic + echo present
        clean_chunk = self.suppress_feedback(read_chunk)

        # Send cleaned chunk
        packed_chunk = self.pack(self.chunk_number, clean_chunk)
        self.send(packed_chunk)

        # Play next received chunk
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)
        return read_chunk


# ------------------------------ Verbose mixin ------------------------------ #
class FeedbackBuffer__verbose(FeedbackBuffer, minimal.Minimal__verbose):
    def __init__(self):
        super().__init__()

    # Reuse stats hooks from Buffering__verbose by counting bytes/packets
    def send(self, packed_chunk):
        buffering_mod.Buffering.send(self, packed_chunk)
        self.sent_bytes_count += len(packed_chunk)
        self.sent_messages_count += 1

    def receive(self):
        packed_chunk = buffering_mod.Buffering.receive(self)
        self.received_bytes_count += len(packed_chunk)
        self.received_messages_count += 1
        return packed_chunk

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if minimal.args.show_samples:
            self.show_recorded_chunk(ADC)
        super()._record_IO_and_play(ADC, DAC, frames, time, status)
        if minimal.args.show_samples:
            self.show_played_chunk(DAC)
        self.recorded_chunk = DAC
        self.played_chunk = ADC

    def _read_IO_and_play(self, DAC, frames, time, status):
        read_chunk = super()._read_IO_and_play(DAC, frames, time, status)
        if minimal.args.show_samples:
            self.show_recorded_chunk(read_chunk)
            self.show_played_chunk(DAC)
        self.recorded_chunk = DAC
        return read_chunk


# --------------------------------- Main ----------------------------------- #
try:
    import argcomplete  # optional tab completion
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")

    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.list_devices:
        import sounddevice as sd
        print("Available devices:")
        print(sd.query_devices())
        raise SystemExit

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = FeedbackBuffer__verbose()
    else:
        intercom = FeedbackBuffer()

    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
