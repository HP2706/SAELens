import re
from typing import List, Tuple, Union

from nnsight import NNsight


def decompose_attr(
    attr: str,
) -> List[Union[str, Tuple[str, List[Union[int, slice, type(...)]]]]]:
    elms = []
    parts = re.split(r"\.", attr)
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

                for dim in idx_content.split(","):
                    dim = dim.strip()
                    if dim == "...":
                        indices.append(...)
                    elif ":" in dim:
                        start, *rest = dim.split(":")
                        start = int(start) if start else None
                        if len(rest) == 1:
                            stop = int(rest[0]) if rest[0] else None
                            indices.append(slice(start, stop))
                        elif len(rest) == 2:
                            stop = int(rest[0]) if rest[0] else None
                            step = int(rest[1]) if rest[1] else None
                            indices.append(slice(start, stop, step))
                    elif dim:
                        indices.append(int(dim))
                    else:
                        indices.append(slice(None))

            elms.append((name, indices))
        else:
            elms.append(part)
    return elms


def get_nested_attr(obj: NNsight, attr_string: str):
    elms = decompose_attr(attr_string)
    for elm in elms:
        if isinstance(elm, tuple):
            name, indices = elm
            obj = getattr(obj, name)
            for idx in indices:
                obj = obj[idx]
        else:
            obj = getattr(obj, elm)
    return obj
