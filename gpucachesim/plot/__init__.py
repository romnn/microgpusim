import matplotlib.colors as mc
import colorsys
import math

from gpucachesim import REPO_ROOT_DIR

PLOT_DIR = REPO_ROOT_DIR / "plot"
TABLE_DIR = REPO_ROOT_DIR / "plot/tables"
EQUATIONS_DIR = REPO_ROOT_DIR / "plot/equations"

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
    "green2": "#a7c957",
    "purple1": "#c21b7b",
    # "blue1": "#196fac",
    # "purple1": "#d62828",
    # "blue1": "#3a86ff",
    "pink1": "#e08dbd",
    "purple2": "#6C22A6",
    "purple3": "#bf7dd5",
    "purple4": "#ac3296",
    "yellow1": "#e8c872ff",
    "yellow2": "#ffbf00",
    "blue1": "#40a2e3ff",
    "red1": "#f24a01ff",
    "red2": "#c60038",
}

RGB_COLOR = {k: hex_to_rgb(v) for k, v in HEX_COLOR.items()}

# SIM_RGB_COLOR = {
#     "simulate": RGB_COLOR["green1"],
#     "execdrivensimulate": RGB_COLOR["yellow2"],
#     "accelsimsimulate": RGB_COLOR["purple4"],
#     # "accelsimsimulate": RGB_COLOR["red2"],
#     "profile": RGB_COLOR["blue1"],
# }

# palette1
SIM_RGB_COLOR = {
    "simulate": hex_to_rgb("#29adb2"),
    "execdrivensimulate": hex_to_rgb("#c5e898"),
    "accelsimsimulate": hex_to_rgb("#0766ad"),
    "profile": hex_to_rgb("#f3f3f3"),
}

# # palette2
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#A2C579"),
#     "execdrivensimulate": hex_to_rgb("#D2DE32"),
#     "accelsimsimulate": hex_to_rgb("#61A3BA"),
#     "profile": hex_to_rgb("#FFFFDD"),
# }

# # palette3
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#427D9D"),
#     "execdrivensimulate": hex_to_rgb("#9BBEC8"),
#     "accelsimsimulate": hex_to_rgb("#164863"),
#     "profile": hex_to_rgb("#DDF2FD"),
# }

# # palette4
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#7091F5"),
#     "execdrivensimulate": hex_to_rgb("#FFFD8C"),
#     "accelsimsimulate": hex_to_rgb("#793FDF"),
#     "profile": hex_to_rgb("#97FFF4"),
# }

# # palette5
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#C23373"),
#     "execdrivensimulate": hex_to_rgb("#F6635C"),
#     "accelsimsimulate": hex_to_rgb("#79155B"),
#     "profile": hex_to_rgb("#FFBA86"),
# }

# # palette6
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#974EC3"),
#     "execdrivensimulate": hex_to_rgb("#504099"),
#     "accelsimsimulate": hex_to_rgb("#313866"),
#     "profile": hex_to_rgb("#FE7BE5"),
# }

# # palette7
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#279EFF"),
#     "execdrivensimulate": hex_to_rgb("#40F8FF"),
#     "accelsimsimulate": hex_to_rgb("#0C356A"),
#     "profile": hex_to_rgb("#D5FFD0"),
# }

# # palette8
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#4942E4"),
#     "execdrivensimulate": hex_to_rgb("#8696FE"),
#     "accelsimsimulate": hex_to_rgb("#11009E"),
#     "profile": hex_to_rgb("#C4B0FF"),
# }

# # palette9
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#6C9BCF"),
#     "execdrivensimulate": hex_to_rgb("#A5C0DD"),
#     "accelsimsimulate": hex_to_rgb("#654E92"),
#     "profile": hex_to_rgb("#EBD8B2"),
# }

# # palette10
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#85CDFD"),
#     "execdrivensimulate": hex_to_rgb("#3C84AB"),
#     "accelsimsimulate": hex_to_rgb("#362FD9"),
#     "profile": hex_to_rgb("#DEFCF9"),
# }

# # palette11
# SIM_RGB_COLOR = {
#     "simulate": hex_to_rgb("#82C3EC"),
#     "execdrivensimulate": hex_to_rgb("#4B56D2"),
#     "accelsimsimulate": hex_to_rgb("#472183"),
#     "profile": hex_to_rgb("#F1F6F5"),
# }

# palette1 > palette2

SIM_MARKER = {
    "simulate": "x",
    "accelsimsimulate": "D",
    "profile": "o",
}

# valid hatches: *+-./OX\ox|
SIM_HATCH = {
    # "simulate": "//",
    # "execdrivensimulate": "xx",
    # "accelsimsimulate": "++",
    # "profile": "xx",
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


def round_to_precision(num: float, precision: int = 2) -> float:
    return round(num, precision)


def round_to_precision_str(num, round_to=2, variable_precision=False):
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
    # handle cases that round to zero but are not actually zero
    if num > 0 and round_to_precision(num, precision=round_to) == 0.0:
        return "<{}{}".format(
            "{:.{}f}".format(1 / 10**round_to, round_to),
            ["", "K", "M", "G", "T", "P"][magnitude],
        )
    if num < 0 and round_to_precision(num, precision=round_to) == 0.0:
        return "<{}{}".format(
            "{:.{}f}".format(1 / 10**round_to, round_to),
            ["", "K", "M", "G", "T", "P"][magnitude],
        )
    return "{}{}".format(
        round_to_precision_str(
            num, round_to=round_to, variable_precision=variable_precision
        ),
        ["", "K", "M", "G", "T", "P", "E", "Z"][magnitude],
    )
