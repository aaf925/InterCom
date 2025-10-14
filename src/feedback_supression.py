import logging
import numpy as np
from scipy import signal, fft
import minimal
import buffer
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

logging.basicConfig(level=logging.INFO)


class Feedback_Suppression(buffer.Buffering):
    """Clase base para supresión de realimentación acústica."""

    def __init__(self):
        super().__init__()
        logging.info("Inicializando Feedback Suppression")

    def _record_io_and_play(self, indata, outdata, frames, time, status):
        """Graba la entrada, aplica supresión y reproduce."""
        # Aplicar la supresión de realimentación
        processed = self.suppress_feedback(indata)
        self.recorded_chunk = processed
        super()._record_io_and_play(processed, outdata, frames, time, status)

    def suppress_feedback(self, indata):
        """Detecta frecuencias dominantes y las atenúa ligeramente."""
        data = indata[:, 0].astype(np.float32)
        spectrum = np.abs(fft.rfft(data))
        freqs = fft.rfftfreq(len(data), 1 / minimal.args.frames_per_second)

        # Detectar picos por encima de un umbral dinámico
        threshold = np.mean(spectrum) * 5
        peak_indices = np.where(spectrum > threshold)[0]

        # Atenuar esas frecuencias (feedback)
        for i in peak_indices:
            start = max(0, i - 1)
            end = min(len(spectrum), i + 1)
            spectrum[start:end] *= 0.2  # atenúa el 80%

        # Reconstruir señal
        suppressed = np.fft.irfft(spectrum).astype(np.int16)
        return suppressed.reshape(-1, 1)


class Feedback_Suppression__verbose(Feedback_Suppression, buffer.Buffering__verbose):
    """Versión con visualización y control interactivo."""

    def __init__(self):
        super().__init__()
        pygame.init()
        self.window_height = 512
        self.display = pygame.display.set_mode((minimal.args.frames_per_chunk // 2, self.window_height))
        pygame.display.set_caption("Feedback Suppression Visualizer")

        # Sliders
        self.threshold_slider = Slider(self.display, 100, 450, 400, 20, min=1, max=10, step=0.5, initial=5)
        self.attenuation_slider = Slider(self.display, 100, 480, 400, 20, min=0.1, max=1.0, step=0.05, initial=0.2)

        # Textboxes
        self.th_text = TextBox(self.display, 520, 450, 60, 30)
        self.at_text = TextBox(self.display, 520, 480, 60, 30)
        self.th_text.disable()
        self.at_text.disable()

    def update_display(self):
        """Visualiza el espectro actual."""
        self.display.fill((0, 0, 0))

        data = self.recorded_chunk[:, 0].astype(np.float32)
        spectrum = np.abs(fft.rfft(data))
        spectrum = np.log1p(spectrum)  # escala logarítmica

        # Dibujar espectro
        width = minimal.args.frames_per_chunk // 2
        normalized = np.interp(spectrum, (0, np.max(spectrum)), (0, self.window_height - 1))
        for x in range(width):
            y = self.window_height - int(normalized[x])
            pygame.draw.line(self.display, (0, 255, 0), (x, self.window_height), (x, y))

        # Actualizar sliders/textos
        self.threshold = self.threshold_slider.getValue()
        self.attenuation = self.attenuation_slider.getValue()
        self.th_text.setText(str(self.threshold))
        self.at_text.setText(f"{self.attenuation:.2f}")

        pygame.display.flip()
        super().update_display()

    def suppress_feedback(self, indata):
        """Versión ajustable con sliders."""
        data = indata[:, 0].astype(np.float32)
        spectrum = np.abs(fft.rfft(data))
        threshold = np.mean(spectrum) * self.threshold
        peak_indices = np.where(spectrum > threshold)[0]

        for i in peak_indices:
            start = max(0, i - 1)
            end = min(len(spectrum), i + 1)
            spectrum[start:end] *= self.attenuation

        suppressed = np.fft.irfft(spectrum).astype(np.int16)
        return suppressed.reshape(-1, 1)


try:
    import argcomplete
except ImportError:
    logging.warning("argcomplete no disponible (opcional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__

    if not any(arg.dest == 'frames_per_chunk' for arg in minimal.parser._actions):
        minimal.parser.add_argument('--frames_per_chunk', type=int, default=1024, help='Número de frames por chunk')

    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete no funcional :-/")

    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_spectrum or minimal.args.show_stats:
        intercom = Feedback_Suppression__verbose()
    else:
        intercom = Feedback_Suppression()

    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
