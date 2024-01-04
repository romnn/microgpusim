import matplotlib.colors as mc
import colorsys
import math

from gpucachesim import REPO_ROOT_DIR

PLOT_DIR = REPO_ROOT_DIR / "plot"

PLOTLY_PDF_OPTS = dict(format="pdf", scale=8)

PPI = 300
FONT_SIZE_PT = 11

# we use the /textwidth from latex as our page width for better font scaling
# DINA4_WIDTH_MM = 210
DINA4_WIDTH_MM = 141.9
DINA4_HEIGHT_MM = 297


def mm_to_inch(mm):
    return mm / 25.4


DINA4_WIDTH_INCHES = mm_to_inch(DINA4_WIDTH_MM)
DINA4_HEIGHT_INCHES = mm_to_inch(DINA4_HEIGHT_MM)


def pt_to_px(pt):
    return int(pt * 4.0 / 3.0)


DINA4_WIDTH = PPI * mm_to_inch(DINA4_WIDTH_MM)
DINA4_HEIGHT = PPI * mm_to_inch(DINA4_HEIGHT_MM)
FONT_SIZE_PX = pt_to_px(FONT_SIZE_PT)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def plotly_rgba(r, g, b, a):
    return "rgba(%d, %d, %d, %f)" % (r, g, b, a)


def plt_rgba(r, g, b, a=1.0):
    return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, a)


HEX_COLOR = {
    "green1": "#81bc4f",
    "purple1": "#c21b7b",
    "blue1": "#196fac",
}

RGB_COLOR = {k: hex_to_rgb(v) for k, v in HEX_COLOR.items()}

SIM_RGB_COLOR = {
    "simulate": RGB_COLOR["green1"],
    "accelsimsimulate": RGB_COLOR["purple1"],
    "profile": RGB_COLOR["blue1"],
}

SIM_MARKER = {
    "simulate": "x",
    "accelsimsimulate": "D",
    "profile": "o",
}

# valid hatches: *+-./OX\ox|
SIM_HATCH = {
    "simulate": "/",
    "accelsimsimulate": "+",
    "profile": "x",
}


def plt_lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plt_darken_color(color, amount=0.5):
    return plt_lighten_color(color, 1.0 + amount)


def round_to_precision(num, round_to=2, variable_precision=False):
    num = round(num, round_to)
    if variable_precision:
        for pos in range(round_to + 1):
            frac, _ = math.modf(num * float(math.pow(10, pos)))
            if frac == 0.0:
                round_to = pos
                break
    return "{:.{}f}".format(num, round_to)


def human_format_thousands(num, round_to=2, variable_precision=False):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = num / 1000.0
    return "{}{}".format(
        round_to_precision(
            num, round_to=round_to, variable_precision=variable_precision
        ),
        ["", "K", "M", "G", "T", "P"][magnitude],
    )
