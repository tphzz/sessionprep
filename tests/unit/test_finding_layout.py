"""Finding headers keep severity before disclosure at both report widths."""
import pytest

from sessionprepgui.detail.report import render_prepared_track_report


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("severity,label", [
    ("problem", "Problem"), ("attention", "Attention"),
    ("information", "Information"), ("clean", "Clean"),
])
def test_severity_precedes_disclosure_and_help_is_not_inline(compact, severity, label):
    row = {"id": "test", "title": "A finding", "severity": severity,
           "summary": "Full result", "fragment": "<tr><td>Details</td></tr>",
           "location": "2 locations", "locations": "Samples 1–2; Samples 3–4",
           "count": "", "hint": "Review the source.", "help": "<p>Explanation only in Help</p>"}
    html = render_prepared_track_report({"header": "File", "rows": [row]}, compact=compact)
    assert html.index(">" + label + "<") < html.index("report-details:test")
    assert html.count(">" + label + "<") == 1
    assert "2 locations" in html and "Review the source." in html
    assert "report-help:test" in html
    assert "Explanation only in Help" not in html


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("severity,label,color", [
    ("problem", "Problem", "#ff4444"), ("attention", "Attention", "#ffaa00"),
    ("information", "Information", "#4499ff"), ("clean", "Clean", "#44cc44"),
])
def test_active_help_preserves_readable_severity_contrast(compact, severity, label, color):
    from html.parser import HTMLParser

    class SeverityBackground(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells = []
            self.background = None

        def handle_starttag(self, tag, attrs):
            if tag == "td":
                styles = dict(attrs).get("style", "")
                value = next((s.split(":", 1)[1].strip() for s in styles.split(";")
                              if s.strip().startswith("background-color:")), None)
                self.cells.append(value)

        def handle_endtag(self, tag):
            if tag == "td":
                self.cells.pop()

        def handle_data(self, value):
            if value.strip() == label:
                self.background = next(bg for bg in reversed(self.cells) if bg)

    row = {"id": "test", "title": "A finding", "severity": severity,
           "summary": "Result", "fragment": "", "location": "2 locations",
           "locations": "", "count": "", "hint": "Review.", "help": "<p>Help</p>"}
    parser = SeverityBackground()
    parser.feed(render_prepared_track_report({"header": "", "rows": [row]},
                                             compact=compact, active_help="test"))

    def luminance(hex_color):
        rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    lighter, darker = sorted((luminance(color), luminance(parser.background)), reverse=True)
    assert (lighter + 0.05) / (darker + 0.05) >= 4.5
