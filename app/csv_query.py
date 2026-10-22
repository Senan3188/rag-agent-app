"""Parse arbitrary CSV schemas and execute bounded, declarative queries (never model code)."""

import csv
import json
import re
from decimal import Decimal, InvalidOperation
from io import StringIO
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.documents import DocumentError


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    operator: Literal["eq", "contains", "gt", "gte", "lt", "lte"] = "eq"
    value: str


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["select", "count", "sum", "avg", "min", "max"] = "select"
    columns: list[str] = Field(default_factory=list, max_length=30)
    filters: list[Filter] = Field(default_factory=list, max_length=10)
    aggregate_column: str | None = None
    sort_by: str | None = None
    descending: bool = False
    limit: int = Field(default=20, ge=1, le=100)
    clarification: str | None = None

    @field_validator("aggregate_column", "sort_by", "clarification", mode="before")
    @classmethod
    def empty_optional(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


def parse_csv(content: bytes):
    try:
        text = content.decode("utf-8-sig")
        try:
            dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(StringIO(text, newline=""), dialect=dialect, strict=True)
        headers = [value.strip() for value in next(reader)]
        if not headers or any(not h for h in headers) or len(set(headers)) != len(headers):
            raise DocumentError("CSV needs nonempty, unique column headers")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise DocumentError(f"CSV record {len(rows) + 2} has a different column count")
            rows.append(dict(zip(headers, row, strict=True)))
        if not rows:
            raise DocumentError("CSV has no data rows")
        return {"columns": headers, "rows": rows}
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise DocumentError("Could not parse CSV; use UTF-8 with a header row") from exc


def number(value):
    try:
        result = Decimal(str(value).strip())
        if not result.is_finite():
            raise InvalidOperation
        return result
    except InvalidOperation as exc:
        raise ValueError(
            "Numeric operation needs plain numeric values; clarify currency or number format"
        ) from exc


def execute(table, plan: QueryPlan):
    names = set(table["columns"])
    requested = [*plan.columns, *(f.column for f in plan.filters)]
    requested += [c for c in (plan.aggregate_column, plan.sort_by) if c is not None]
    if any(c not in names for c in requested):
        raise ValueError("Query references an unknown column; please clarify the column name")

    def matches(row, condition):
        value = row[condition.column].strip().casefold()
        target = condition.value.strip().casefold()
        if condition.operator == "eq":
            return value == target
        if condition.operator == "contains":
            return target in value
        left, right = number(value), number(target)
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[
            condition.operator
        ]

    rows = [
        (i + 2, row)
        for i, row in enumerate(table["rows"])
        if all(matches(row, f) for f in plan.filters)
    ]
    total = len(rows)
    if plan.operation == "count":
        return {"operation": "count", "matched_rows": total, "value": total}
    if plan.operation != "select":
        if plan.aggregate_column is None:
            raise ValueError("An aggregate column is required")
        values = [
            number(row[plan.aggregate_column])
            for _, row in rows
            if row[plan.aggregate_column].strip()
        ]
        value = None
        if values:
            value = {
                "sum": lambda: sum(values),
                "avg": lambda: sum(values) / len(values),
                "min": lambda: min(values),
                "max": lambda: max(values),
            }[plan.operation]()
        return {
            "operation": plan.operation,
            "column": plan.aggregate_column,
            "matched_rows": total,
            "numeric_values": len(values),
            "value": str(value) if value is not None else None,
        }
    if plan.sort_by:
        key = plan.sort_by
        try:
            numeric = {i: number(row[key]) for i, row in rows}
        except ValueError:
            numeric = None
        rows.sort(
            key=lambda pair: numeric[pair[0]] if numeric is not None else pair[1][key].casefold(),
            reverse=plan.descending,
        )
    columns = plan.columns or table["columns"]
    output = [
        {"csv_record": i, "values": {c: row[c] for c in columns}} for i, row in rows[: plan.limit]
    ]
    result = {
        "operation": "select",
        "matched_rows": total,
        "returned_rows": len(output),
        "truncated": total > len(output),
        "rows": output,
    }
    # Never silently send an incomplete serialized table to the answer model.
    if len(json.dumps(result)) > 24000:
        raise ValueError("Result is too large; request fewer rows or columns")
    return result


def referenced_values(question, table):
    """Find exact and partial cell references across every column, including IDs."""
    normalized = question.casefold()
    tokens = re.findall(r"\w+", normalized)
    stop = {
        "what",
        "which",
        "where",
        "who",
        "the",
        "and",
        "for",
        "with",
        "from",
        "show",
        "give",
        "list",
        "does",
        "have",
        "that",
        "this",
        "all",
        "are",
        "how",
        "many",
    }
    header_tokens = {t for c in table["columns"] for t in re.findall(r"\w+", c.casefold())}
    found = {}
    for row in table["rows"]:
        for column, raw in row.items():
            value = raw.strip()
            if not value or len(value) > 240:
                continue
            exact = bool(
                re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", normalized)
            )
            partial = None
            if not exact:
                cell_tokens = re.findall(r"\w+", value.casefold())
                for width in range(min(6, len(tokens)), 0, -1):
                    for i in range(len(tokens) - width + 1):
                        phrase = tokens[i : i + width]
                        if not any(len(t) >= 3 and t not in stop | header_tokens for t in phrase):
                            continue
                        if any(
                            cell_tokens[j : j + width] == phrase
                            for j in range(len(cell_tokens) - width + 1)
                        ):
                            partial = " ".join(phrase)
                            break
                    if partial:
                        break
            if exact or partial:
                item = found.setdefault(
                    value.casefold(),
                    {
                        "value": value,
                        "columns": [],
                        "operator": "eq" if exact else "contains",
                        "matched_text": value if exact else partial,
                    },
                )
                if column not in item["columns"]:
                    item["columns"].append(column)
    # A complete referenced cell takes precedence over fragments inside it.
    exact_values = [h["value"].casefold() for h in found.values() if h["operator"] == "eq"]
    filtered = [
        h
        for h in found.values()
        if not any(
            h["matched_text"].casefold() != value and h["matched_text"].casefold() in value
            for value in exact_values
        )
    ]
    return sorted(filtered, key=lambda h: (h["operator"] != "eq", -len(h["matched_text"])))[:30]


def matching_samples(table, hints):
    """Return bounded representative rows containing question-referenced values."""
    pairs = {(c, h["value"].casefold()) for h in hints for c in h["columns"]}
    rows = [
        r for r in table["rows"] if any((c, v.strip().casefold()) in pairs for c, v in r.items())
    ]
    return [{c: v[:240] for c, v in row.items()} for row in (rows or table["rows"])[:3]]


def ground_filters(plan, table, hints):
    """Correct an absent exact value only when its column is unambiguous.

    Do not broaden comparisons, alter values, or guess when multiple columns match.
    """
    columns = set(table["columns"])
    grounded = []
    for condition in plan.filters:
        corrected = condition
        if condition.operator == "eq" and condition.column in columns:
            target = condition.value.strip().casefold()
            hint = next((h for h in hints if h["value"].casefold() == target), None)
            exists = any(
                row[condition.column].strip().casefold() == target for row in table["rows"]
            )
            if not exists and hint and len(hint["columns"]) == 1:
                corrected = condition.model_copy(update={"column": hint["columns"][0]})
        grounded.append(corrected)
    return plan.model_copy(update={"filters": grounded})


def lookup_evidence(question, table):
    """Retrieve complete matching records for simple questions without generated filters.

    Comparisons, aggregates and ordering keep the structured execution path.
    Candidate rows are evidence, not a claim that every match satisfies the question.
    """
    if re.search(
        r"\b(how many|how much|count|total|sum|average|mean|median|minimum|maximum|"
        r"highest|lowest|greater|less|over|under|before|after|between|top|bottom|sort|"
        r"order|older|younger|at least|at most|more than|equal|group|distinct|unique)\b|[<>]",
        question,
        re.IGNORECASE,
    ):
        return None
    hints = referenced_values(question, table)
    if not hints:
        return None
    matches = []
    for index, row in enumerate(table["rows"], 2):
        fields = [
            c
            for hint in hints
            for c in hint["columns"]
            if row[c].strip().casefold() == hint["value"].casefold()
        ]
        if fields:
            matches.append(
                {"csv_record": index, "matched_columns": list(dict.fromkeys(fields)), "values": row}
            )
    output = []
    for row in matches[:100]:
        if len(json.dumps(output + [row])) > 20000:
            break
        output.append(row)
    if not output:
        return None
    return {
        "operation": "lookup_candidates",
        "references": hints,
        "matched_rows": len(matches),
        "returned_rows": len(output),
        "truncated": len(output) < len(matches),
        "rows": output,
    }
