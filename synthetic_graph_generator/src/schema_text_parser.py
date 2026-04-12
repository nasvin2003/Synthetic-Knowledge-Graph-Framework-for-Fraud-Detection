import json
import re
from pathlib import Path
from typing import Any, Dict, List


_NODE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(?:\[(.*)\])?\s*$")
_REL_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:->|→)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*(?:\{(.*)\})?\s*(?:\[(.*)\])?\s*$"
)


def _strip_comments(line: str) -> str:
    for marker in ("#", "//"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line.strip()


def _split_top_level(text: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth_round = depth_square = depth_curly = 0
    in_string = False
    string_char = ""

    for ch in text:
        if in_string:
            buf.append(ch)
            if ch == string_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            buf.append(ch)
            continue

        if ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1

        if ch == delimiter and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None

    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value):
            return float(value)
    except ValueError:
        pass

    return value


def _parse_keyvals(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not text or not text.strip():
        return result

    for token in _split_top_level(text):
        if "=" not in token:
            raise ValueError(f"Expected key=value pair, got: {token}")
        key, value = token.split("=", 1)
        result[key.strip()] = _coerce_value(value)
    return result


def _normalize_prop_type(type_name: str) -> str:
    mapping = {
        "integer": "int",
        "int": "int",
        "long": "int",
        "float": "float",
        "double": "double",
        "bool": "bool",
        "boolean": "bool",
        "str": "string",
        "string": "string",
    }
    return mapping.get(type_name.strip().lower(), type_name.strip().lower())


def _parse_property_token(token: str) -> Dict[str, Any]:
    token = token.strip()
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)(?:\[(.*)\])?", token)
    if not m:
        raise ValueError(f"Invalid property definition: {token}")

    name, type_name, opts_text = m.groups()
    prop: Dict[str, Any] = {
        "name": name,
        "type": _normalize_prop_type(type_name),
    }
    if opts_text:
        prop.update(_parse_keyvals(opts_text))
    return prop


def _parse_node_line(line: str) -> Dict[str, Any]:
    m = _NODE_RE.fullmatch(line)
    if not m:
        raise ValueError(f"Invalid node declaration: {line}")

    label, props_text, opts_text = m.groups()
    node: Dict[str, Any] = {
        "label": label,
        "properties": [],
    }

    props_text = (props_text or "").strip()
    if props_text:
        node["properties"] = [_parse_property_token(tok) for tok in _split_top_level(props_text)]

    opts = _parse_keyvals(opts_text or "")
    node.update(opts)
    return node


def _assign_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = target
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _parse_relationship_line(line: str) -> Dict[str, Any]:
    m = _REL_RE.fullmatch(line)
    if not m:
        raise ValueError(f"Invalid relationship declaration: {line}")

    rel_type, from_node, to_node, props_text, opts_text = m.groups()
    rel: Dict[str, Any] = {
        "type": rel_type,
        "from_node": from_node,
        "to_node": to_node,
        "properties": [],
        "constraints": {},
    }

    props_text = (props_text or "").strip()
    if props_text:
        rel["properties"] = [_parse_property_token(tok) for tok in _split_top_level(props_text)]

    opts = _parse_keyvals(opts_text or "")
    for key, value in opts.items():
        if key == "count":
            rel["count"] = value
        elif key == "allow_self_loop":
            rel.setdefault("constraints", {})["allow_self_loop"] = value
        elif key.startswith("from_degree."):
            _assign_nested(rel.setdefault("constraints", {}), key, value)
        elif key.startswith("to_degree."):
            _assign_nested(rel.setdefault("constraints", {}), key, value)
        elif key.startswith("constraints."):
            _assign_nested(rel, key, value)
        else:
            rel[key] = value

    return rel


def _split_schema_statements(text: str) -> List[str]:
    statements: List[str] = []
    buf: List[str] = []
    depth_round = depth_square = depth_curly = 0
    in_string = False
    string_char = ""

    for raw_line in text.splitlines():
        line = _strip_comments(raw_line)
        if not line:
            continue

        if not buf and line.startswith("@"):
            statements.append(line)
            continue

        if buf:
            buf.append(" ")
        buf.append(line)

        for ch in line:
            if in_string:
                if ch == string_char:
                    in_string = False
                continue

            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                continue

            if ch == "(":
                depth_round += 1
            elif ch == ")":
                depth_round -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
            elif ch == "{":
                depth_curly += 1
            elif ch == "}":
                depth_curly -= 1

        if depth_round == 0 and depth_square == 0 and depth_curly == 0 and buf:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []

    if in_string or depth_round != 0 or depth_square != 0 or depth_curly != 0:
        raise ValueError("Unbalanced schema declaration, check parentheses/brackets/braces in the text schema.")

    if buf:
        stmt = "".join(buf).strip()
        if stmt:
            statements.append(stmt)

    return statements


def parse_pseudo_graph_schema(text: str) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "seed": 0,
        "no_duplicate_triples": True,
        "nodes": [],
        "relationships": [],
    }

    statements = _split_schema_statements(text)
    for idx, stmt in enumerate(statements, start=1):
        if stmt.startswith("@"):
            directive = stmt[1:]
            if "=" not in directive:
                raise ValueError(f"Statement {idx}: invalid directive: {stmt}")
            key, value = directive.split("=", 1)
            schema[key.strip()] = _coerce_value(value)
            continue

        if "->" in stmt or "→" in stmt:
            schema["relationships"].append(_parse_relationship_line(stmt))
        else:
            schema["nodes"].append(_parse_node_line(stmt))

    return schema


def load_schema_definition(schema_source: str) -> Dict[str, Any]:
    path = Path(schema_source)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()

    if path.suffix.lower() == ".json" or stripped.startswith("{"):
        return json.loads(text)
    return parse_pseudo_graph_schema(text)


def save_pseudo_schema_as_json(schema_source: str, output_json_path: str) -> Dict[str, Any]:
    schema = load_schema_definition(schema_source)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return schema
