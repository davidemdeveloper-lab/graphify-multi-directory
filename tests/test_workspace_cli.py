from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


PYTHON = sys.executable


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def _write_queryable_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    G = nx.Graph()
    G.add_node("alpha", label="Alpha", file_type="code", source_file="a.py")
    G.add_node("beta", label="Beta", file_type="code", source_file="b.py")
    G.add_edge("alpha", "beta", relation="calls", confidence="EXTRACTED", source_file="a.py", weight=1.0)
    try:
        data = json_graph.node_link_data(G, edges="links")
    except TypeError:
        data = json_graph.node_link_data(G)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_workspace_cli_init_add_source_path_and_doctor(tmp_path):
    env = os.environ.copy()
    env["GRAPHIFY_WORKSPACES_DIR"] = str(tmp_path / "workspaces")
    source_dir = tmp_path / "api"
    source_dir.mkdir()

    init = _run(["workspace", "init", "product"], tmp_path, env)
    assert init.returncode == 0, init.stderr

    add = _run(["workspace", "add-source", "product", "api", str(source_dir), "--kind", "service"], tmp_path, env)
    assert add.returncode == 0, add.stderr

    path = _run(["workspace", "path", "product"], tmp_path, env)
    assert path.returncode == 0, path.stderr
    assert path.stdout.strip().endswith("workspaces/product/graphify-out/graph.json")

    shutil.rmtree(source_dir)
    doctor = _run(["workspace", "doctor", "product"], tmp_path, env)
    assert doctor.returncode != 0
    assert "missing source api" in doctor.stderr.lower()


def test_workspace_cli_add_list_and_remove_relation(tmp_path):
    env = os.environ.copy()
    env["GRAPHIFY_WORKSPACES_DIR"] = str(tmp_path / "workspaces")
    api = tmp_path / "api"
    docs = tmp_path / "docs"
    api.mkdir()
    docs.mkdir()

    assert _run(["workspace", "init", "product"], tmp_path, env).returncode == 0
    assert _run(["workspace", "add-source", "product", "api", str(api)], tmp_path, env).returncode == 0
    assert _run(["workspace", "add-source", "product", "docs", str(docs), "--kind", "docs"], tmp_path, env).returncode == 0

    add = _run(
        [
            "workspace",
            "add-relation",
            "product",
            "api",
            "docs",
            "--relation",
            "documents",
            "--confidence",
            "inferred",
            "--confidence-score",
            "0.7",
            "--description",
            "API is documented by docs",
        ],
        tmp_path,
        env,
    )

    assert add.returncode == 0, add.stderr
    assert "api --documents--> docs" in add.stdout

    listed = _run(["workspace", "list-relations", "product"], tmp_path, env)
    assert listed.returncode == 0, listed.stderr
    assert "api --documents--> docs [INFERRED]" in listed.stdout

    removed = _run(
        ["workspace", "remove-relation", "product", "api", "docs", "--relation", "documents"],
        tmp_path,
        env,
    )

    assert removed.returncode == 0, removed.stderr
    assert "Removed 1 relation(s)" in removed.stdout

    listed = _run(["workspace", "list-relations", "product"], tmp_path, env)
    assert listed.returncode == 0, listed.stderr
    assert "No relations" in listed.stdout


def test_query_uses_workspace_pointer_when_no_graph_flag_or_env(tmp_path):
    env = os.environ.copy()
    env.pop("GRAPHIFY_OUT", None)
    workspace_dir = tmp_path / "workspaces" / "product"
    graph_path = workspace_dir / "graphify-out" / "graph.json"
    _write_queryable_graph(graph_path)

    repo = tmp_path / "repo"
    repo.mkdir()
    pointer_dir = repo / ".graphify"
    pointer_dir.mkdir()
    (pointer_dir / "workspace.json").write_text(
        json.dumps({
            "workspace": "product",
            "manifest_path": str(workspace_dir / "workspace.json"),
            "graph_path": str(graph_path),
        }),
        encoding="utf-8",
    )

    result = _run(["query", "Alpha"], repo, env)

    assert result.returncode == 0, result.stderr
    assert "Alpha" in result.stdout


def test_workspace_build_code_source_writes_central_graph_and_report(tmp_path):
    env = os.environ.copy()
    env["GRAPHIFY_WORKSPACES_DIR"] = str(tmp_path / "workspaces")
    source_dir = tmp_path / "api"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("def hello():\n    return 'ok'\n", encoding="utf-8")

    assert _run(["workspace", "init", "product"], tmp_path, env).returncode == 0
    add = _run(["workspace", "add-source", "product", "api", str(source_dir), "--kind", "service"], tmp_path, env)
    assert add.returncode == 0, add.stderr

    build = _run(["workspace", "build", "product"], tmp_path, env)

    assert build.returncode == 0, build.stderr
    out = tmp_path / "workspaces" / "product" / "graphify-out"
    assert (out / "graph.json").exists()
    assert (out / "GRAPH_REPORT.md").exists()
    data = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in data["nodes"]}
    assert any(node_id.startswith("api::") for node_id in node_ids)


def test_workspace_cli_rejects_invalid_max_workers_without_traceback(tmp_path):
    env = os.environ.copy()
    env["GRAPHIFY_WORKSPACES_DIR"] = str(tmp_path / "workspaces")

    init = _run(["workspace", "init", "product"], tmp_path, env)
    assert init.returncode == 0, init.stderr

    result = _run(["workspace", "build", "product", "--max-workers", "abc"], tmp_path, env)

    assert result.returncode == 2
    assert "argument --max-workers" in result.stderr
    assert "Traceback" not in result.stderr


def test_workspace_cli_shows_workspace_help(tmp_path):
    env = os.environ.copy()

    result = _run(["workspace", "--help"], tmp_path, env)

    assert result.returncode == 0
    assert "usage: graphify workspace" in result.stdout
    assert "add-source" in result.stdout


def test_workspace_cli_shows_subcommand_help(tmp_path):
    env = os.environ.copy()

    result = _run(["workspace", "build", "--help"], tmp_path, env)

    assert result.returncode == 0
    assert "usage: graphify workspace build" in result.stdout
    assert "--max-workers" in result.stdout


def test_workspace_cli_rejects_unknown_flags_without_ignoring_them(tmp_path):
    env = os.environ.copy()
    env["GRAPHIFY_WORKSPACES_DIR"] = str(tmp_path / "workspaces")

    init = _run(["workspace", "init", "product"], tmp_path, env)
    assert init.returncode == 0, init.stderr

    result = _run(["workspace", "build", "product", "--bogus"], tmp_path, env)

    assert result.returncode == 2
    assert "unrecognized arguments: --bogus" in result.stderr
