import platform
import pyperclip
import numpy as np


def flatten(l):
    return [item for ll in l for item in ll]


def dedup(l):
    return list(dict.fromkeys(l))


def two_closest_divisors(n):
    assert isinstance(n, int)
    a = int(np.round(np.sqrt(n)))
    while n % a > 0:
        a -= 1
    return a, n // a


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
    try:
        pyperclip.copy(value)
    except pyperclip.PyperclipException as e:
        print("copy to clipboard failed: {}".format(e))
