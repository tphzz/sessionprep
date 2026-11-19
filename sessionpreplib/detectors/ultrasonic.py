from ..detector import TrackDetector
from ..models import DetectorResult, IssueLocation, Severity
from ..repairs import band_evidence


class UltrasonicDetector(TrackDetector):
    id = "ultrasonic"
    name = "Ultrasonic Content"
    shorthand = "US"

    @classmethod
    def html_help(cls):
        return ("Checks for sustained ultrasonic energy from 24.1 kHz to Nyquist on each channel. "
                "Five consecutive 100 ms windows must exceed both −60 dBFS RMS and "
                "−35 dB relative to program energy. The source must be at least 50 kHz.")

    def analyze(self, track):
        if track.samplerate < 50000:
            return DetectorResult(self.id, Severity.CLEAN, "sample rate below 50 kHz",
                                  {"eligible": False, "regions": []}, applicable=False)
        evidence = band_evidence(
            track.audio_data, track.samplerate, low_hz=24100, high_hz=track.samplerate / 2,
            window_sec=0.1, level_db=-60, ratio_db=-35, consecutive=5,
        )
        issues = [
            IssueLocation(start, end, channel, Severity.INFO, self.id,
                          "Sustained ultrasonic energy", 24100, track.samplerate / 2)
            for start, end, channel in evidence["regions"]
        ]
        return DetectorResult(
            self.id, Severity.INFO if issues else Severity.CLEAN,
            "sustained ultrasonic content" if issues else "no sustained ultrasonic content",
            {"eligible": True, **evidence}, issues=issues,
            hint="review ultrasonic filtering" if issues else None,
        )
