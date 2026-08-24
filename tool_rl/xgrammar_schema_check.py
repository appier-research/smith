"""
check_xgrammar_schema.py

Validates whether a JSON schema is compatible with xgrammar's
JSON-schema-to-EBNF converter. Reports known incompatible patterns
so you can fix them before hitting a RuntimeError in vLLM/SGLang.

Usage:
    python check_xgrammar_schema.py schema.json
    python check_xgrammar_schema.py '{"type": "object", ...}'

Or as a library:
    from check_xgrammar_schema import check_schema
    result = check_schema({"type": "object", ...})
    print(result)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# ── Known problematic patterns ──────────────────────────────────────────────

ARRAY_UNSUPPORTED_KEYWORDS = {
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
}

# minItems / maxItems were unsupported in older xgrammar (< 0.1.30-ish).
# They are now partially supported, but we still flag them as a warning.
ARRAY_WARN_KEYWORDS = {
    "minItems",
    "maxItems",
}

STRING_UNSUPPORTED_KEYWORDS = {
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
}

OBJECT_UNSUPPORTED_KEYWORDS = {
    "dependentRequired",
    "dependentSchemas",
}

# Draft-04 style "items" as array is the exact trigger for
#   RuntimeError: items must be a boolean or an object
DRAFT04_ITEMS_ARRAY_MSG = (
    '"items" is an array (Draft-04 tuple validation). '
    "xgrammar expects items to be a boolean or object. "
    'Migrate to "prefixItems" (Draft 2020-12) instead.'
)


@dataclass
class SchemaCheckResult:
    """Aggregated result of all checks."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        lines: list[str] = []
        if self.ok and not self.warnings:
            lines.append("✅ Schema looks compatible with xgrammar.")
        else:
            if self.errors:
                lines.append(f"❌ Found {len(self.errors)} error(s):")
                for i, e in enumerate(self.errors, 1):
                    lines.append(f"   {i}. {e}")
            if self.warnings:
                lines.append(f"⚠️  Found {len(self.warnings)} warning(s):")
                for i, w in enumerate(self.warnings, 1):
                    lines.append(f"   {i}. {w}")
            if not self.errors:
                lines.insert(0, "✅ No hard errors, but check warnings below.")
        return "\n".join(lines)


# ── Recursive checker ────────────────────────────────────────────────────────

def _check_node(
    node: dict,
    path: str,
    result: SchemaCheckResult,
    defs: dict | None = None,
    visited_refs: set | None = None,
) -> None:
    """Recursively inspect a schema node for xgrammar-incompatible patterns."""
    if not isinstance(node, dict):
        return

    if visited_refs is None:
        visited_refs = set()

    # Resolve $defs / definitions at root level
    if defs is None:
        defs = node.get("$defs", node.get("definitions", {}))

    # ── $ref resolution (basic) ──────────────────────────────────────────
    if "$ref" in node:
        ref = node["$ref"]
        if ref in visited_refs:
            return  # avoid infinite recursion on circular refs
        visited_refs = visited_refs | {ref}
        if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
            key = ref.split("/")[-1]
            if key in defs:
                _check_node(defs[key], f"{path}/$ref({key})", result, defs, visited_refs)
        return

    schema_type = node.get("type")

    # ── Array checks ─────────────────────────────────────────────────────
    if schema_type == "array" or "items" in node or "prefixItems" in node:
        # THE critical check: "items" must be bool or object, NOT an array
        items = node.get("items")
        if isinstance(items, list):
            result.errors.append(f"{path}: {DRAFT04_ITEMS_ARRAY_MSG}")
        elif items is not None and not isinstance(items, (dict, bool)):
            result.errors.append(
                f'{path}: "items" has unexpected type {type(items).__name__}. '
                "Expected a boolean or object."
            )

        # Check for unsupported array keywords
        for kw in ARRAY_UNSUPPORTED_KEYWORDS:
            if kw in node:
                result.errors.append(
                    f'{path}: "{kw}" is not supported by xgrammar.'
                )
        for kw in ARRAY_WARN_KEYWORDS:
            if kw in node:
                result.warnings.append(
                    f'{path}: "{kw}" may not be supported in older xgrammar versions. '
                    "Upgrade to >= 0.1.30 or verify your version."
                )

        # Recurse into items / prefixItems / unevaluatedItems
        if isinstance(items, dict):
            _check_node(items, f"{path}/items", result, defs, visited_refs)
        for i, sub in enumerate(node.get("prefixItems", [])):
            _check_node(sub, f"{path}/prefixItems[{i}]", result, defs, visited_refs)
        unevaluated = node.get("unevaluatedItems")
        if isinstance(unevaluated, dict):
            _check_node(unevaluated, f"{path}/unevaluatedItems", result, defs, visited_refs)

    # ── Object checks ────────────────────────────────────────────────────
    if schema_type == "object" or "properties" in node:
        for kw in OBJECT_UNSUPPORTED_KEYWORDS:
            if kw in node:
                result.errors.append(
                    f'{path}: "{kw}" is not supported by xgrammar.'
                )

        # Recurse into properties
        for prop_name, prop_schema in node.get("properties", {}).items():
            _check_node(
                prop_schema, f"{path}/properties/{prop_name}", result, defs, visited_refs
            )

        # additionalProperties
        ap = node.get("additionalProperties")
        if isinstance(ap, dict):
            _check_node(ap, f"{path}/additionalProperties", result, defs, visited_refs)

        # patternProperties
        for pat, pat_schema in node.get("patternProperties", {}).items():
            _check_node(
                pat_schema, f'{path}/patternProperties/"{pat}"', result, defs, visited_refs
            )

    # ── String checks ────────────────────────────────────────────────────
    if schema_type == "string":
        for kw in STRING_UNSUPPORTED_KEYWORDS:
            if kw in node:
                result.warnings.append(
                    f'{path}: "{kw}" is not supported by xgrammar.'
                )

    # ── Composition keywords ─────────────────────────────────────────────
    for comp_kw in ("anyOf", "oneOf", "allOf"):
        comp_val = node.get(comp_kw)
        # JSON Schema allows boolean schemas (e.g. {"anyOf": true}); guard against
        # non-list values so enumerate() doesn't raise TypeError.
        if not isinstance(comp_val, list):
            continue
        for i, sub in enumerate(comp_val):
            _check_node(sub, f"{path}/{comp_kw}[{i}]", result, defs, visited_refs)

    if_schema = node.get("if")
    if isinstance(if_schema, dict):
        result.warnings.append(
            f'{path}: "if/then/else" is not well supported by xgrammar.'
        )

    not_schema = node.get("not")
    if isinstance(not_schema, dict):
        result.warnings.append(
            f'{path}: "not" keyword has limited support in xgrammar.'
        )


# ── Public API ───────────────────────────────────────────────────────────────

def check_schema(schema: Union[dict, str]) -> SchemaCheckResult:
    """
    Check whether a JSON schema is compatible with xgrammar.

    Args:
        schema: A dict or a JSON string representing the schema.

    Returns:
        SchemaCheckResult with errors and warnings.
    """
    if isinstance(schema, str):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        r = SchemaCheckResult()
        r.errors.append(f"Top-level schema must be an object, got {type(schema).__name__}.")
        return r

    result = SchemaCheckResult()
    _check_node(schema, "#", result)
    return result


# ── Optional: live xgrammar test ─────────────────────────────────────────────

def check_schema_live(schema: Union[dict, str]) -> SchemaCheckResult:
    import xgrammar as xgr
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    """
    Try to actually compile the schema with xgrammar (if installed).
    Falls back to static checks if xgrammar is not available.
    """
    result = check_schema(schema)
    schema_str = json.dumps(schema) if isinstance(schema, dict) else schema

    # Use a small tokenizer for testing
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    try:
        compiler.compile_json_schema(schema_str)
        if not result.ok:
            result.warnings.append(
                "Static analysis found issues, but xgrammar compiled OK. "
                "Your xgrammar version may have fixed these."
            )
    except Exception as e:
        err_msg = str(e)
        result.errors.append(f"xgrammar compilation failed: {err_msg}")


    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

TEST_CASES: list[tuple[str, dict]] = [
    (
        "Draft-04 items as array (triggers 'items must be a boolean or an object')",
        {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "array",
                    "items": [{"type": "number"}, {"type": "number"}],
                }
            },
        },
    ),
    (
        "Fixed version using prefixItems (Draft 2020-12)",
        {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "array",
                    "prefixItems": [{"type": "number"}, {"type": "number"}],
                    "items": False,
                }
            },
        },
    ),
    (
        "Array with minItems / maxItems (older xgrammar warning)",
        {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
    ),
    (
        "Unsupported: uniqueItems + contains",
        {
            "type": "array",
            "items": {"type": "integer"},
            "uniqueItems": True,
            "contains": {"type": "integer", "minimum": 5},
        },
    ),
    (
        "Simple valid object schema",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
    ),
    (
        "Nested issue via $defs / $ref",
        {
            "type": "object",
            "properties": {
                "point": {"$ref": "#/$defs/Point"},
            },
            "$defs": {
                "Point": {
                    "type": "array",
                    "items": [{"type": "number"}, {"type": "number"}],
                }
            },
        },
    ),
    (
        "Tool-calling schema with minItems (vLLM auto-generated)",
        {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    ),
    (
        "if/then/else (limited support)",
        {
            "type": "object",
            "properties": {"kind": {"type": "string"}},
            "if": {"properties": {"kind": {"const": "a"}}},
            "then": {"properties": {"value": {"type": "integer"}}},
            "else": {"properties": {"value": {"type": "string"}}},
        },
    ),
    (
        "Deeply nested items-as-array inside oneOf",
        {
            "type": "object",
            "properties": {
                "data": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": [{"type": "string"}, {"type": "integer"}],
                        },
                    ]
                }
            },
        },
    ),
]
 
 
def main() -> None:
    agreed = 0
    disagreed = 0

    for i, (name, schema) in enumerate(TEST_CASES, 1):
        static_result = check_schema(schema)
        live_result = check_schema_live(schema)

        static_ok = static_result.ok
        # live_result includes static errors plus any xgrammar compilation error,
        # so we only care about whether xgrammar itself accepted or rejected it.
        live_compilation_ok = not any(
            e.startswith("xgrammar compilation failed") for e in live_result.errors
        )

        agrees = static_ok == live_compilation_ok
        if agrees:
            agreed += 1
            agreement_label = "AGREE"
        else:
            disagreed += 1
            agreement_label = "DISAGREE"

        print(f"{'─' * 60}")
        print(f"Test {i}: {name}")
        print(f"Schema: {json.dumps(schema, indent=2)}")
        print(f"  Static check : {'OK' if static_ok else 'FAIL'}")
        print(f"  Live compile : {'OK' if live_compilation_ok else 'FAIL'}")
        print(f"  Agreement    : [{agreement_label}]")
        if not agrees:
            if static_ok and not live_compilation_ok:
                print("  -> Static check passed but xgrammar rejected the schema.")
                live_errors = [e for e in live_result.errors if e.startswith("xgrammar compilation failed")]
                for e in live_errors:
                    print(f"     {e}")
            else:
                print("  -> Static check flagged errors but xgrammar compiled successfully.")
                print(f"     Static errors: {static_result.errors}")
        print()

    print(f"{'═' * 60}")
    print(f"Summary: {agreed} agreed, {disagreed} disagreed, {len(TEST_CASES)} total")
 


if __name__ == "__main__":
    main()