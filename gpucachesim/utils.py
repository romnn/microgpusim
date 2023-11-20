import platform
import pyperclip
import numpy as np


def flatten(l):
    return [item for ll in l for item in ll]


def round_up_to_next_power_of_two(x):
    exp = np.ceil(np.log2(x)) if x > 0 else 1
    return np.power(2, exp)


def round_down_to_next_power_of_two(x):
    exp = np.floor(np.log2(x)) if x > 0 else 1
    return np.power(2, exp)


def round_to_multiple_of(x, multiple_of):
    return multiple_of * np.round(x / multiple_of)


def round_up_to_multiple_of(x, multiple_of):
    return multiple_of * np.ceil(x / multiple_of)


def round_down_to_multiple_of(x, multiple_of):
    return multiple_of * np.floor(x / multiple_of)


def copy_to_clipboard(value):
    pyperclip.copy(value)
    # tk.clipboard_clear()
    # tk.clipboard_append(value)
    # p = platform.system()
    # match p:
    #     case "Linux":
    #         command = 'echo "{}" | clip'.format(value.strip())
    #     case "Darwin":
    #         command = 'echo "{}" | pbcopy'.format(value.strip())
    #     case other:
    #         raise ValueError("cannot copy to clipboard on platform {}".format(other))
    # os.system(command)
