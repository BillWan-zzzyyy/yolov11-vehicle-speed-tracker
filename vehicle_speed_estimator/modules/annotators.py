import supervision as sv
from config.settings import SPEEDING_LABEL_BG_COLOR_HEX, NORMAL_LABEL_BG_COLOR_HEX

def get_annotators(fps):
    colors = ("#007fff", "#0072e6", "#0066cc", "#0059b3", "#004c99", "#004080", "#003366", "#00264d")
    palette = sv.ColorPalette([sv.Color.from_hex(c) for c in colors])
    speeding_palette = sv.ColorPalette([sv.Color.from_hex(SPEEDING_LABEL_BG_COLOR_HEX)])
    normal_palette = sv.ColorPalette([sv.Color.from_hex(NORMAL_LABEL_BG_COLOR_HEX)])
    return {
        "bbox": sv.BoxAnnotator(color=palette, thickness=2, color_lookup=sv.ColorLookup.TRACK),
        "trace": sv.TraceAnnotator(color=palette, trace_length=fps, color_lookup=sv.ColorLookup.TRACK),
        "label": sv.RichLabelAnnotator(color=normal_palette, font_size=16, color_lookup=sv.ColorLookup.INDEX),
        "speeding_label": sv.RichLabelAnnotator(color=speeding_palette, font_size=16, color_lookup=sv.ColorLookup.INDEX),
    }
