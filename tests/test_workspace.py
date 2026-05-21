from __future__ import annotations

import json
import shutil
from pathlib import Path

import networkx as nx
import pytest
from networkx.readwrite import json_graph


def _write_graph(path: Path, node_id: str, label: str, source_file: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    G = nx.Graph()
    G.add_node(node_id, label=label, file_type="code", source_file=source_file)
    try:
        data = json_graph.node_link_data(G, edges="links")
    except TypeError:
        data = json_graph.node_link_data(G)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_init_workspace_creates_manifest(monkeypatch, tmp_path):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")

    manifest = workspace.init_workspace("product")

    manifest_path = tmp_path / "workspaces" / "product" / "workspace.json"
    assert manifest_path.exists()
    assert manifest["name"] == "product"
    assert manifest["sources"] == {}
    assert Path(manifest["graph_path"]) == tmp_path / "workspaces" / "product" / "graphify-out" / "graph.json"


@pytest.mark.parametrize("bad_id", [".", "..", "..."])
def test_workspace_rejects_dot_only_names(monkeypatch, tmp_path, bad_id):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")

    with pytest.raises(workspace.WorkspaceError):
        workspace.init_workspace(bad_id)


@pytest.mark.parametrize("bad_id", [".", "..", "..."])
def test_workspace_rejects_dot_only_source_ids(monkeypatch, tmp_path, bad_id):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    workspace.init_workspace("knowledge")

    with pytest.raises(workspace.WorkspaceError):
        workspace.add_source("knowledge", bad_id, source_dir)


def test_add_source_stores_generic_source_and_pointer(monkeypatch, tmp_path):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    workspace.init_workspace("knowledge")
    manifest = workspace.add_source("knowledge", "research_docs", source_dir, kind="research")

    source = manifest["sources"]["research_docs"]
    assert source["kind"] == "research"
    assert source["path"] == str(source_dir.resolve())

    pointer = source_dir / ".graphify" / "workspace.json"
    pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
    assert pointer_data["workspace"] == "knowledge"
    assert Path(pointer_data["manifest_path"]) == tmp_path / "workspaces" / "knowledge" / "workspace.json"


def test_doctor_reports_missing_source_without_searching(monkeypatch, tmp_path):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")
    source_dir = tmp_path / "api"
    source_dir.mkdir()

    workspace.init_workspace("product")
    workspace.add_source("product", "api", source_dir, kind="service")
    shutil.rmtree(source_dir)

    status = workspace.doctor_workspace("product")

    assert status["ok"] is False
    assert status["missing_sources"] == [{"id": "api", "path": str(source_dir.resolve())}]


def test_doctor_reports_missing_workspace_outputs(monkeypatch, tmp_path, capsys):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")
    source_dir = tmp_path / "api"
    source_dir.mkdir()

    workspace.init_workspace("product")
    workspace.add_source("product", "api", source_dir, kind="service")

    status = workspace.doctor_workspace("product")
    rc = workspace.print_doctor(status)
    captured = capsys.readouterr()

    assert status["ok"] is False
    assert status["graph_exists"] is False
    assert status["missing_source_graphs"] == [
        {
            "id": "api",
            "path": str(tmp_path / "workspaces" / "product" / "sources" / "api" / "graphify-out" / "graph.json"),
        }
    ]
    assert rc == 1
    assert "workspace graph missing" in captured.err
    assert "source graph missing api" in captured.err


def test_doctor_reports_source_path_that_is_not_directory(monkeypatch, tmp_path):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")
    source_file = tmp_path / "api"
    source_file.write_text("not a directory", encoding="utf-8")

    manifest = workspace.init_workspace("product")
    manifest["sources"] = {
        "api": {
            "id": "api",
            "path": str(source_file),
            "kind": "service",
            "label": "API",
        }
    }
    workspace.save_workspace(manifest)

    status = workspace.doctor_workspace("product")

    assert status["ok"] is False
    assert status["non_directory_sources"] == [{"id": "api", "path": str(source_file)}]


def test_compose_workspace_graph_prefixes_nodes_and_keeps_source_metadata(tmp_path):
    from graphify.workspace import compose_workspace_graph

    api_graph = tmp_path / "api" / "graph.json"
    docs_graph = tmp_path / "docs" / "graph.json"
    _write_graph(api_graph, "shared", "Shared", "src/shared.py")
    _write_graph(docs_graph, "shared", "Shared", "guide.md")

    manifest_path = tmp_path / "workspace.json"
    manifest = {
        "name": "product",
        "manifest_path": str(manifest_path),
        "sources": {
            "api": {"id": "api", "path": str((tmp_path / "api-src").resolve()), "kind": "service", "label": "API"},
            "docs": {"id": "docs", "path": str((tmp_path / "docs-src").resolve()), "kind": "docs", "label": "Docs"},
        },
        "relations": [
            {"source": "api", "target": "docs", "relation": "documents", "description": "API is documented by docs"}
        ],
    }

    G = compose_workspace_graph(manifest, {"api": api_graph, "docs": docs_graph})

    assert "api::shared" in G.nodes
    assert "docs::shared" in G.nodes
    assert G.nodes["api::shared"]["workspace"] == "product"
    assert G.nodes["api::shared"]["source_id"] == "api"
    assert G.nodes["docs::shared"]["source_kind"] == "docs"
    assert G.has_edge("source::api", "source::docs")
    assert G.edges["source::api", "source::docs"]["relation"] == "documents"


def test_workspace_allows_local_ollama_without_api_key(monkeypatch):
    from graphify import workspace

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    assert workspace.resolve_semantic_backend("ollama") == "ollama"


def test_workspace_partial_update_reuses_stored_detection_metadata(monkeypatch, tmp_path):
    from graphify import workspace

    monkeypatch.setattr(workspace, "_WORKSPACES_DIR", tmp_path / "workspaces")
    api = tmp_path / "api"
    docs = tmp_path / "docs"
    api.mkdir()
    docs.mkdir()

    manifest = workspace.init_workspace("product")
    manifest["sources"] = {
        "api": {
            "id": "api",
            "path": str(api),
            "kind": "service",
            "label": "API",
            "last_detection": {"files": {"code": ["a.py"]}, "total_files": 1, "total_words": 10},
            "last_tokens": {"input": 1, "output": 2},
        },
        "docs": {
            "id": "docs",
            "path": str(docs),
            "kind": "docs",
            "label": "Docs",
            "last_detection": {"files": {"document": ["guide.md"]}, "total_files": 1, "total_words": 20},
            "last_tokens": {"input": 3, "output": 4},
        },
    }
    workspace.save_workspace(manifest)

    detections, tokens = workspace.collect_workspace_report_inputs(
        workspace.load_workspace("product")
    )

    assert set(detections) == {"api", "docs"}
    assert tokens == {"input": 4, "output": 6}
