import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.csv_query import QueryPlan, execute, parse_csv
from app.documents import DocumentError
from app.main import create_app
from app.store import Store


class CSVBackend:
    def __init__(self, plan):
        self.plan = QueryPlan.model_validate(plan)
        self.evidence = None

    async def embed(self, texts):
        raise AssertionError("CSV must not use vector retrieval")

    async def plan_csv(self, question, table):
        assert "columns" in table
        return self.plan

    async def chat(self, messages):
        self.evidence = json.loads(json.loads(messages[1]["content"])["evidence"])
        return "Sergio's last name is Silva [1]."


def test_csv_api_searches_last_row_and_persists(tmp_path):
    backend = CSVBackend(
        {"columns": ["Given", "Family"], "filters": [{"column": "Given", "value": "Sergio"}]}
    )
    settings = Settings(data_dir=tmp_path)
    data = "Given;Family\n" + "Other;Person\n" * 100 + "Sergio;Silva\n"
    with TestClient(create_app(settings, backend)) as api:
        response = api.post("/documents", files={"file": ("people.csv", data.encode())})
        assert response.status_code == 201
        doc = response.json()
        assert doc["embedding_model"] == "structured-csv"
    with TestClient(create_app(settings, backend)) as api:
        response = api.post(f"/documents/{doc['id']}/ask", json={"question": "Sergio surname?"})
        assert response.status_code == 200, response.text
        assert response.json()["sources"][0]["kind"] == "csv_result"
        result = backend.evidence["result"]
        assert result["matched_rows"] == 1
        assert result["rows"][0]["csv_record"] == 102
        assert result["rows"][0]["values"]["Family"] == "Silva"
        api.delete(f"/documents/{doc['id']}")
        assert Store(tmp_path).csv_table(doc["id"]) is None


def test_different_schema_and_quoted_values():
    table = parse_csv(
        b'Product,Unit Price,Description\nWidget,12.5,"a,b"\nGadget,20,"two\nlines"\n'
    )
    result = execute(table, QueryPlan(operation="sum", aggregate_column="Unit Price"))
    assert result["value"] == "32.5"
    assert table["rows"][1]["Description"] == "two\nlines"


@pytest.mark.parametrize("data", [b"A,A\n1,2", b",B\n1,2", b"A,B\n1", b"A,B\n", b"\xff"])
def test_invalid_csv(data):
    with pytest.raises(DocumentError):
        parse_csv(data)


def test_case_insensitive_exact_matching_and_duplicates():
    table = parse_csv(b"Name,Surname\nSergio,Silva\nsergio,Lopez\nSergio Jr,Jones\n")
    result = execute(table, QueryPlan(filters=[{"column": "Name", "value": " SERGIO "}]))
    assert result["matched_rows"] == 2
    assert [r["values"]["Surname"] for r in result["rows"]] == ["Silva", "Lopez"]


def test_count_and_aggregate_use_all_rows_before_limit():
    table = parse_csv(b"Name,Cost\na,2\nb,10\nc,30\n")
    assert execute(table, QueryPlan(operation="count", limit=1))["value"] == 3
    assert (
        execute(table, QueryPlan(operation="avg", aggregate_column="Cost", limit=1))["value"]
        == "14"
    )
    result = execute(table, QueryPlan(sort_by="Cost", descending=True, limit=1))
    assert result["truncated"]
    assert result["rows"][0]["values"]["Cost"] == "30"


def test_numeric_filter_and_no_match():
    table = parse_csv(b"ID,Amount\nx,4\ny,20\n")
    plan = QueryPlan(filters=[{"column": "Amount", "operator": "gt", "value": "10"}])
    assert execute(table, plan)["rows"][0]["values"]["ID"] == "y"
    plan = QueryPlan(filters=[{"column": "ID", "value": "absent"}])
    assert execute(table, plan)["matched_rows"] == 0


def test_unknown_column_and_non_numeric_values_rejected():
    table = parse_csv(b"Name,Cost\na,$12\n")
    with pytest.raises(ValueError, match="unknown column"):
        execute(table, QueryPlan(columns=["invented"]))
    with pytest.raises(ValueError, match="plain numeric"):
        execute(table, QueryPlan(operation="sum", aggregate_column="Cost"))
    with pytest.raises(ValueError):
        QueryPlan.model_validate({"operation": "select", "python": "print('unsafe')"})


def test_old_csv_gives_reupload_message(tmp_path):
    store = Store(tmp_path)
    doc = store.add("old.csv", "old", [{"index": 0, "page": None, "text": "Name\nx"}], [[1]])
    with TestClient(create_app(Settings(data_dir=tmp_path), CSVBackend({}))) as api:
        response = api.post(f"/documents/{doc['id']}/ask", json={"question": "names?"})
        assert response.status_code == 409
        assert "Re-upload" in response.json()["detail"]


def test_clarification_has_no_query_results(tmp_path):
    backend = CSVBackend({"clarification": "Which amount column should I use?"})
    with TestClient(create_app(Settings(data_dir=tmp_path), backend)) as api:
        doc = api.post("/documents", files={"file": ("x.csv", b"A,B\n1,2")}).json()
        response = api.post(f"/documents/{doc['id']}/ask", json={"question": "total?"})
        assert response.status_code == 200
        assert response.json()["sources"] == []
        assert backend.evidence is None


def test_csv_document_isolation(tmp_path):
    backend = CSVBackend({"filters": [{"column": "Name", "value": "Sergio"}]})
    with TestClient(create_app(Settings(data_dir=tmp_path), backend)) as api:
        first = api.post(
            "/documents", files={"file": ("a.csv", b"Name,Family\nSergio,Silva")}
        ).json()
        api.post("/documents", files={"file": ("b.csv", b"Name,Family\nSergio,Other")})
        response = api.post(f"/documents/{first['id']}/ask", json={"question": "Sergio?"})
        assert response.status_code == 200
        assert backend.evidence["result"]["matched_rows"] == 1
        assert backend.evidence["result"]["rows"][0]["values"]["Family"] == "Silva"


def test_structured_planner_adapter():
    import asyncio

    import httpx

    from app.ollama import Ollama, OllamaError

    async def run():
        backend = Ollama(Settings())
        await backend.client.aclose()

        def respond(request):
            payload = json.loads(request.content)
            assert payload["format"]["additionalProperties"] is False
            assert payload["model"] == "qwen3:8b"
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps({"filters": [{"column": "Name", "value": "Sergio"}]})
                    }
                },
            )

        backend.client = httpx.AsyncClient(
            base_url="http://ollama", transport=httpx.MockTransport(respond)
        )
        try:
            plan = await backend.plan_csv("Sergio?", parse_csv(b"Name,Family\nSergio,Silva"))
            assert plan.filters[0].value == "Sergio"
            backend.client = await replace_client(backend.client)
            with pytest.raises(OllamaError):
                await backend.plan_csv("Sergio?", parse_csv(b"Name,Family\nSergio,Silva"))
        finally:
            await backend.client.aclose()

    async def replace_client(client):
        await client.aclose()
        return httpx.AsyncClient(
            base_url="http://ollama",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"message": {"content": '{"operation":"delete"}'}}
                )
            ),
        )

    asyncio.run(run())


def test_empty_optional_model_fields():
    table = parse_csv(b"ID,Name\n1,A\n2,B")
    plan = QueryPlan(operation="count", sort_by="", aggregate_column="", clarification="")
    assert plan.sort_by is None
    assert execute(table, plan)["value"] == 2


def test_wrong_name_column_is_grounded_in_actual_data():
    from app.csv_query import ground_filters, referenced_values

    table = parse_csv(b'First Name,Last Name,Company\nBarry,Spence,"Farrell, Rubio and Newman"\n')
    question = "What is Barry's Company ?"
    plan = QueryPlan(columns=["Company"], filters=[{"column": "Last Name", "value": "Barry"}])
    grounded = ground_filters(plan, table, referenced_values(question, table))
    assert grounded.filters[0].column == "First Name"
    result = execute(table, grounded)
    assert result["rows"][0]["values"]["Company"] == "Farrell, Rubio and Newman"


def test_grounding_handles_different_columns_and_unicode_possessive():
    from app.csv_query import ground_filters, referenced_values

    table = parse_csv(b"Title,Brand,Price\nWidget,Acme,12\nWidget,Other,20\n")
    hints = referenced_values("What is Acme’s Price?", table)
    plan = QueryPlan(filters=[{"column": "Title", "value": "Acme"}])
    grounded = ground_filters(plan, table, hints)
    result = execute(table, grounded)
    assert result["matched_rows"] == 1
    assert result["rows"][0]["values"]["Price"] == "12"


def test_grounding_does_not_guess_ambiguous_values_or_partial_words():
    from app.csv_query import ground_filters, referenced_values

    table = parse_csv(b"Given,Family,Alias\nBarry,Spence,Barry\nCarl,Smith,Other")
    plan = QueryPlan(filters=[{"column": "Family", "value": "Barry"}])
    assert ground_filters(plan, table, referenced_values("Barry's company?", table)) == plan
    assert referenced_values("Carlos's company?", table) == []
    assert referenced_values("Unknown person's company?", table) == []


def test_grounding_preserves_valid_filters_and_duplicates():
    from app.csv_query import ground_filters, referenced_values

    table = parse_csv(b"Given,Family\nBarry,Spence\nBarry,Smith\nOther,Barry")
    plan = QueryPlan(filters=[{"column": "Family", "value": "Barry"}])
    assert ground_filters(plan, table, referenced_values("Family Barry?", table)) == plan
    given_plan = QueryPlan(filters=[{"column": "Given", "value": "Barry"}])
    assert (
        execute(table, ground_filters(given_plan, table, referenced_values("Barry?", table)))[
            "matched_rows"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("Company", "Acme Holdings"),
        ("City", "New Town"),
        ("Email", "person@example.com"),
        ("Phone", "+1-555-1234"),
        ("Customer Code", "008"),
        ("Index", "7"),
        ("Subscription Date", "2025-06-01"),
        ("Region", "BE"),
    ],
)
def test_references_are_discovered_in_any_column(column, value):
    from app.csv_query import matching_samples, referenced_values

    table = {
        "columns": ["Unrelated", column],
        "rows": [
            {"Unrelated": "first", column: "other"},
            {"Unrelated": "target", column: value},
        ],
    }
    hints = referenced_values(f"What is Unrelated for {value}?", table)
    assert any(h["value"] == value and h["columns"] == [column] for h in hints)
    assert matching_samples(table, hints)[0]["Unrelated"] == "target"


def test_partial_company_and_longer_cell_references():
    from app.csv_query import referenced_values

    table = parse_csv(b'Organization,Owner\n"Farrell, Rubio and Newman",Example\n')
    hints = referenced_values("Who belongs to Rubio?", table)
    assert hints[0]["value"] == "Farrell, Rubio and Newman"
    assert hints[0]["operator"] == "contains"
    assert hints[0]["matched_text"] == "rubio"


def test_planner_repairs_zero_results_using_execution_feedback():
    import asyncio

    import httpx

    from app.ollama import Ollama

    async def run():
        backend = Ollama(Settings())
        await backend.client.aclose()
        calls = []

        def respond(request):
            payload = json.loads(request.content)
            calls.append(payload)
            if len(calls) == 1:
                plan = {
                    "columns": ["Owner"],
                    "filters": [{"column": "Organization", "value": "Rubio"}],
                }
            else:
                assert "execution_feedback" in payload["messages"][-1]["content"]
                plan = {
                    "columns": ["Owner"],
                    "filters": [
                        {"column": "Organization", "operator": "contains", "value": "Rubio"}
                    ],
                }
            return httpx.Response(200, json={"message": {"content": json.dumps(plan)}})

        backend.client = httpx.AsyncClient(
            base_url="http://ollama", transport=httpx.MockTransport(respond)
        )
        try:
            table = parse_csv(b'Organization,Owner\n"Farrell, Rubio and Newman",Example\n')
            plan = await backend.plan_csv("Who belongs to Rubio?", table)
            assert len(calls) == 2
            assert execute(table, plan)["matched_rows"] == 1
        finally:
            await backend.client.aclose()

    asyncio.run(run())


def test_unresolved_query_does_not_claim_matching_data_is_absent():
    import asyncio

    import httpx

    from app.ollama import Ollama

    async def run():
        backend = Ollama(Settings())
        await backend.client.aclose()
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps(
                            {
                                "columns": ["Owner"],
                                "filters": [{"column": "Organization", "value": "Rubio"}],
                            }
                        )
                    }
                },
            )

        backend.client = httpx.AsyncClient(
            base_url="http://ollama", transport=httpx.MockTransport(respond)
        )
        try:
            table = parse_csv(b'Organization,Owner\n"Farrell, Rubio and Newman",Example\n')
            plan = await backend.plan_csv("Who belongs to Rubio?", table)
            assert len(calls) == 2
            assert plan.clarification
        finally:
            await backend.client.aclose()

    asyncio.run(run())


def test_full_reference_suppresses_fragments_in_other_columns():
    from app.csv_query import matching_samples, referenced_values

    table = parse_csv(
        b'Company,Name,Website\n"Farrell, Rubio and Newman",Barry,example.com\nOther,Farrell,rubio.com\n'
    )
    hints = referenced_values("Who works at Farrell, Rubio and Newman?", table)
    assert len(hints) == 1
    assert hints[0]["columns"] == ["Company"]
    assert matching_samples(table, hints)[0]["Name"] == "Barry"


@pytest.mark.parametrize("value", ["Acme", "New Town", "person@example.com", "008", "2025-06-01"])
def test_simple_lookup_uses_full_rows_without_planner(tmp_path, value):
    class LookupBackend(CSVBackend):
        async def plan_csv(self, question, table):
            raise AssertionError("Simple value lookups must not depend on generated filters")

    backend = LookupBackend({})
    data = "Organization,Location,Contact,Code,Date,Answer\nAcme,New Town,person@example.com,008,2025-06-01,found\n"
    with TestClient(create_app(Settings(data_dir=tmp_path), backend)) as api:
        doc = api.post("/documents", files={"file": ("arbitrary.csv", data.encode())}).json()
        response = api.post(
            f"/documents/{doc['id']}/ask", json={"question": f"What is Answer for {value}?"}
        )
        assert response.status_code == 200
        assert backend.evidence["result"]["operation"] == "lookup_candidates"
        assert backend.evidence["result"]["rows"][0]["values"]["Answer"] == "found"


def test_partial_lookup_preserves_ambiguous_matches():
    from app.csv_query import lookup_evidence

    table = parse_csv(
        b'Company,Name,City\n"Kelly, Hahn and Williams",Derek,Port Kirkfurt\n"Roman, Ross and Kelly",Mercedes,North Brianborough\n'
    )
    result = lookup_evidence("What is Kelly's City?", table)
    assert result["matched_rows"] == 2
    assert all(r["matched_columns"] == ["Company"] for r in result["rows"])
    assert {r["values"]["Name"] for r in result["rows"]} == {"Derek", "Mercedes"}


def test_lookup_does_not_override_aggregates_or_numeric_filters():
    from app.csv_query import lookup_evidence

    table = parse_csv(b"Product,Price\nWidget,12\n")
    for question in [
        "How many Widget products?",
        "Widget price greater than 10?",
        "Sum of Widget Price?",
    ]:
        assert lookup_evidence(question, table) is None
