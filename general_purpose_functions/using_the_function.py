from typing import Any


def f_value_in_center_of_segment(func: Any, segment: Any):
    return func(0.5*(segment[1]+segment[0]))