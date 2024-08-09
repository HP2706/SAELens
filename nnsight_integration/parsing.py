import re
from typing import Any, List, Tuple, Union

import numpy as np
from nnsight import NNsight


def process_idx(dim: str) -> int | list | slice:
    print("dim", dim)
    if dim == "...":
        return Ellipsis  # This is the correct way to return the ... object
    elif ":" in dim:
        start, stop, step = (None, None, None)
        parts = dim.split(":")
        if len(parts) > 0 and parts[0]:
            start = int(parts[0])
        if len(parts) > 1 and parts[1]:
            stop = int(parts[1])
        if len(parts) > 2 and parts[2]:
            step = int(parts[2])
        return slice(start, stop, step)
    elif dim:
        return int(dim)
    else:
        return slice(None)


def decompose_attr(
    attr: str,
) -> List[Union[str, Tuple[str, List[Union[int, slice, type(...)]]]]]:
    elms = []
    # Use a regex that doesn't split on dots inside brackets
    parts = re.findall(r"(\w+(?:\[.*?\])*)", attr)
    for part in parts:
        match = re.match(r"(\w+)(\[.+\])?", part)
        if match and match.group(2):
            name = match.group(1)
            bracket_content = match.group(2)
            indices = []
            while bracket_content:
                bracket_match = re.match(r"\[([^\[\]]*)\](.*)", bracket_content)
                if not bracket_match:
                    raise ValueError(f"Invalid bracket format in: {part}")
                idx_content = bracket_match.group(1)
                bracket_content = bracket_match.group(2)
                if "," in idx_content:
                    inner_idx = []
                    for dim in idx_content.split(","):
                        dim = dim.strip()
                        inner_idx.append(process_idx(dim))
                    indices.append(inner_idx)
                else:
                    indices.append(process_idx(idx_content))
            elms.append((name, indices))
        else:
            elms.append(part)
    return elms


def get_nested_attr(obj: NNsight, attr_string: str) -> Any:
    elms = decompose_attr(attr_string)
    for elm in elms:
        if isinstance(elm, tuple):
            name, indices = elm
            obj = getattr(obj, name)
            print("indices", indices)
            for idx in indices:
                if isinstance(idx, list):
                    # Handle multi-dimensional indexing
                    obj = obj[tuple(idx)]
                else:
                    obj = obj[idx]
        else:
            print(f"attribute elm: {elm}")
            obj = getattr(obj, elm)
    return obj
