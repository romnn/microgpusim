def human_readable(n) -> str:
    # return "{:}".format(n)

    import math

    if abs(float(int(n)) - float(n)) == 0:
        n = int(n)

    precision = 5
    if isinstance(n, float):
        res = "{:.5f}".format(n)
    else:
        res = "{}".format(n)
    # before the point
    parts = res.split(".")
    before, after = parts[0], "".join(parts[1:])

    num_segments = int(math.ceil(float(len(before)) / 3.0))
    before = reversed([before[max(len(before) - (3 * i) - 3, 0) : len(before) - (3 * i)] for i in range(num_segments)])
    before = " ".join(before)
    if after == "":
        # print(before.replace(" ", ""))
        # print(f"{round(n, precision):f}")
        assert float(before.replace(" ", "")) == round(n, precision)
        return before

    # print((before + "." + after).replace(" ", ""))
    # print(f"{round(n, precision):f}")
    assert float((before + "." + after).replace(" ", "")) == round(n, precision)
    return before + "." + after
    # return "{:,}".format(n).replace(",", " ")
