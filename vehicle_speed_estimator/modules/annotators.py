import supervision as sv
from config.settings import (
    ABNORMAL_LABEL_BG_COLOR_HEX,
    DETECTION_LABEL_FONT_SIZE,
    NORMAL_BBOX_TRACE_COLOR_HEX,
    NORMAL_LABEL_BG_COLOR_HEX,
)

def get_annotators(fps):
    bbox_trace_palette = sv.ColorPalette([sv.Color.from_hex(NORMAL_BBOX_TRACE_COLOR_HEX)])
    abnormal_palette = sv.ColorPalette([sv.Color.from_hex(ABNORMAL_LABEL_BG_COLOR_HEX)])
    normal_palette = sv.ColorPalette([sv.Color.from_hex(NORMAL_LABEL_BG_COLOR_HEX)])
    return {
        "bbox": sv.BoxAnnotator(color=bbox_trace_palette, thickness=2, color_lookup=sv.ColorLookup.INDEX),
        "trace": sv.TraceAnnotator(color=bbox_trace_palette, trace_length=fps, color_lookup=sv.ColorLookup.INDEX),
        "label": sv.RichLabelAnnotator(
            color=normal_palette,
            font_size=DETECTION_LABEL_FONT_SIZE,
            color_lookup=sv.ColorLookup.INDEX
        ),
        "abnormal_label": sv.RichLabelAnnotator(
            color=abnormal_palette,
            font_size=DETECTION_LABEL_FONT_SIZE,
            color_lookup=sv.ColorLookup.INDEX
        ),
    }
