from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


_WORKSPACES_DIR = Path(
    os.environ.get("GRAPHIFY_WORKSPACES_DIR", Path.home() / ".graphify" / "workspaces")
)
_POINTER_PATH = Path(".graphify") / "workspace.json"
_VALID_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_VALID_RELATION = re.compile(r"^[A-Za-z0-9_.:-]+$")
_VALID_CONFIDENCE = {"EXTRACTED", "INFERRED", "AMBIGUOUS"}


class WorkspaceError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_id(value: str, label: str) -> str:
    value = value.strip()
    if not value or not _VALID_ID.match(value) or "::" in value or set(value) == {"."}:
        raise WorkspaceError(f"invalid {label}: {value!r}")
    return value


def _validate_relation(value: str | None) -> str:
    relation = (value or "relates_to").strip()
    if not relation or not _VALID_RELATION.match(relation):
        raise WorkspaceError(f"invalid relation type: {value!r}")
    return relation


def _validate_confidence(value: str | None) -> str:
    confidence = (value or "EXTRACTED").strip().upper()
    if confidence not in _VALID_CONFIDENCE:
        allowed = ", ".join(sorted(_VALID_CONFIDENCE))
        raise WorkspaceError(f"invalid relation confidence: {value!r}; expected one of {allowed}")
    return confidence


def _validate_score(value: float | int | str | None, label: str) -> float:
    if value is None:
        return 1.0
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkspaceError(f"invalid {label}: {value!r}") from exc
    if score < 0 or score > 1:
        raise WorkspaceError(f"invalid {label}: {value!r}; expected 0..1")
    return score


def _validate_weight(value: float | int | str | None) -> float:
    if value is None:
        return 1.0
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkspaceError(f"invalid weight: {value!r}") from exc
    if weight < 0:
        raise WorkspaceError(f"invalid weight: {value!r}; expected a non-negative number")
    return weight


def resolve_semantic_backend(backend: str | None = None) -> str:
    from graphify.semantic_backend import (
        SemanticBackendError,
        resolve_semantic_backend as _resolve_semantic_backend,
    )

    try:
        return _resolve_semantic_backend(backend)
    except SemanticBackendError as exc:
        raise WorkspaceError(str(exc)) from exc


def workspace_dir(name: str) -> Path:
    return _WORKSPACES_DIR / _validate_id(name, "workspace name")


def manifest_path(name: str) -> Path:
    return workspace_dir(name) / "workspace.json"


def _graph_path_for(name: str) -> Path:
    return workspace_dir(name) / "graphify-out" / "graph.json"


def _source_graph_path(name: str, source_id: str) -> Path:
    return workspace_dir(name) / "sources" / source_id / "graphify-out" / "graph.json"


def _manifest_for(name: str) -> dict[str, Any]:
    wd = workspace_dir(name)
    return {
        "version": 1,
        "name": name,
        "manifest_path": str((wd / "workspace.json").resolve()),
        "graph_path": str((wd / "graphify-out" / "graph.json").resolve()),
        "sources": {},
        "relations": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


def load_workspace(name: str) -> dict[str, Any]:
    path = manifest_path(name)
    if not path.exists():
        raise WorkspaceError(f"workspace not found: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("manifest_path", str(path.resolve()))
    data.setdefault("graph_path", str(_graph_path_for(data.get("name", name)).resolve()))
    data.setdefault("sources", {})
    data.setdefault("relations", [])
    return data


def save_workspace(manifest: dict[str, Any]) -> dict[str, Any]:
    name = _validate_id(str(manifest.get("name", "")), "workspace name")
    path = manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = str(path.resolve())
    manifest.setdefault("graph_path", str(_graph_path_for(name).resolve()))
    manifest.setdefault("sources", {})
    manifest.setdefault("relations", [])
    manifest["updated_at"] = _now()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def init_workspace(name: str) -> dict[str, Any]:
    name = _validate_id(name, "workspace name")
    path = manifest_path(name)
    if path.exists():
        return load_workspace(name)
    manifest = _manifest_for(name)
    Path(manifest["graph_path"]).parent.mkdir(parents=True, exist_ok=True)
    return save_workspace(manifest)


def write_pointer(source_dir: Path, manifest: dict[str, Any]) -> Path:
    pointer = source_dir / _POINTER_PATH
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "version": 1,
                "workspace": manifest["name"],
                "manifest_path": manifest["manifest_path"],
                "graph_path": manifest["graph_path"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pointer


def add_source(
    name: str,
    source_id: str,
    path: str | Path,
    *,
    kind: str = "folder",
    label: str | None = None,
) -> dict[str, Any]:
    manifest = load_workspace(name)
    source_id = _validate_id(source_id, "source id")
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise WorkspaceError(f"source path not found: {source_path}")
    if not source_path.is_dir():
        raise WorkspaceError(f"source path is not a directory: {source_path}")
    source = {
        "id": source_id,
        "path": str(source_path),
        "kind": kind or "folder",
        "label": label or source_id,
        "last_seen": _now(),
    }
    sources = manifest.setdefault("sources", {})
    previous_source = sources.get(source_id)
    sources[source_id] = source
    manifest = save_workspace(manifest)
    try:
        write_pointer(source_path, manifest)
    except OSError as exc:
        if previous_source is None:
            sources.pop(source_id, None)
        else:
            sources[source_id] = previous_source
        save_workspace(manifest)
        raise WorkspaceError(f"could not write workspace pointer for source {source_id}: {exc}") from exc
    return manifest


def add_relation(
    name: str,
    source_id: str,
    target_id: str,
    *,
    relation: str = "relates_to",
    confidence: str = "EXTRACTED",
    confidence_score: float | int | str | None = 1.0,
    description: str | None = None,
    weight: float | int | str | None = 1.0,
) -> dict[str, Any]:
    manifest = load_workspace(name)
    source_id = _validate_id(source_id, "source id")
    target_id = _validate_id(target_id, "target source id")
    sources = manifest.setdefault("sources", {})
    if source_id not in sources:
        raise WorkspaceError(f"source not found in workspace: {source_id}")
    if target_id not in sources:
        raise WorkspaceError(f"source not found in workspace: {target_id}")
    if source_id == target_id:
        raise WorkspaceError("workspace relation endpoints must be different sources")

    relation = _validate_relation(relation)
    entry: dict[str, Any] = {
        "source": source_id,
        "target": target_id,
        "relation": relation,
        "confidence": _validate_confidence(confidence),
        "confidence_score": _validate_score(confidence_score, "confidence_score"),
        "weight": _validate_weight(weight),
    }
    if description:
        entry["description"] = description

    relations = manifest.setdefault("relations", [])
    manifest["relations"] = [
        r
        for r in relations
        if not isinstance(r, dict)
        or not (
            r.get("source") == source_id
            and r.get("target") == target_id
            and r.get("relation", "relates_to") == relation
        )
    ]
    manifest["relations"].append(entry)
    return save_workspace(manifest)


def remove_relation(
    name: str,
    source_id: str,
    target_id: str,
    *,
    relation: str | None = None,
) -> tuple[dict[str, Any], int]:
    manifest = load_workspace(name)
    source_id = _validate_id(source_id, "source id")
    target_id = _validate_id(target_id, "target source id")
    relation = _validate_relation(relation) if relation else None
    relations = manifest.setdefault("relations", [])
    kept = [
        r
        for r in relations
        if not isinstance(r, dict)
        or not (
            r.get("source") == source_id
            and r.get("target") == target_id
            and (relation is None or r.get("relation", "relates_to") == relation)
        )
    ]
    removed = len(relations) - len(kept)
    manifest["relations"] = kept
    return save_workspace(manifest), removed


def list_relations(name: str) -> list[dict[str, Any]]:
    manifest = load_workspace(name)
    return list(manifest.get("relations", []))


def doctor_workspace(name: str) -> dict[str, Any]:
    manifest = load_workspace(name)
    missing: list[dict[str, str]] = []
    non_directory: list[dict[str, str]] = []
    missing_source_graphs: list[dict[str, str]] = []
    invalid_relations: list[dict[str, str]] = []
    source_ids = set(manifest.get("sources", {}))
    for source_id, source in manifest.get("sources", {}).items():
        path = Path(source.get("path", "")).expanduser()
        if not path.exists():
            missing.append({"id": source_id, "path": str(path)})
        elif not path.is_dir():
            non_directory.append({"id": source_id, "path": str(path)})
        source_graph = _source_graph_path(manifest["name"], source_id)
        if not source_graph.exists():
            missing_source_graphs.append({"id": source_id, "path": str(source_graph)})
    for index, relation in enumerate(manifest.get("relations", [])):
        if not isinstance(relation, dict):
            invalid_relations.append(
                {
                    "index": str(index),
                    "source": "",
                    "target": "",
                    "problem": "relation entry is not an object",
                }
            )
            continue
        src = str(relation.get("source", ""))
        tgt = str(relation.get("target", ""))
        problems = []
        if src not in source_ids:
            problems.append(f"missing source {src!r}")
        if tgt not in source_ids:
            problems.append(f"missing target {tgt!r}")
        try:
            _validate_relation(str(relation.get("relation", "relates_to")))
            _validate_confidence(str(relation.get("confidence", "EXTRACTED")))
            _validate_score(relation.get("confidence_score", 1.0), "confidence_score")
            _validate_weight(relation.get("weight", 1.0))
        except WorkspaceError as exc:
            problems.append(str(exc))
        if problems:
            invalid_relations.append(
                {
                    "index": str(index),
                    "source": src,
                    "target": tgt,
                    "problem": "; ".join(problems),
                }
            )
    graph_path = Path(manifest.get("graph_path", ""))
    return {
        "ok": (
            not missing
            and not non_directory
            and graph_path.exists()
            and not missing_source_graphs
            and not invalid_relations
        ),
        "workspace": manifest["name"],
        "manifest_path": manifest["manifest_path"],
        "graph_path": str(graph_path),
        "graph_exists": graph_path.exists(),
        "missing_sources": missing,
        "non_directory_sources": non_directory,
        "missing_source_graphs": missing_source_graphs,
        "invalid_relations": invalid_relations,
    }


def find_workspace_pointer(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    candidates = [cur] + list(cur.parents)
    for base in candidates:
        pointer = base / _POINTER_PATH
        if pointer.exists():
            return pointer
    return None


def resolve_workspace_graph_path(start: Path | None = None) -> Path | None:
    pointer = find_workspace_pointer(start)
    if pointer is None:
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:
        return None
    graph_path = data.get("graph_path")
    if graph_path:
        return Path(graph_path).expanduser()
    manifest_path_raw = data.get("manifest_path")
    if manifest_path_raw:
        manifest_path_obj = Path(manifest_path_raw).expanduser()
        if manifest_path_obj.exists():
            try:
                manifest = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
                if manifest.get("graph_path"):
                    return Path(manifest["graph_path"]).expanduser()
            except Exception:
                return None
    return None


def _load_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text(encoding="utf-8"))
    links = data.get("links", data.get("edges"))
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise WorkspaceError(f"source graph is not node-link JSON: {path}")

    graph = nx.Graph()
    for node in nodes:
        if not isinstance(node, dict) or "id" not in node:
            raise WorkspaceError(f"source graph has invalid node in {path}")
        graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in links:
        if not isinstance(edge, dict) or "source" not in edge or "target" not in edge:
            raise WorkspaceError(f"source graph has invalid edge in {path}")
        src = edge["source"]
        tgt = edge["target"]
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        attrs["_src"] = src
        attrs["_tgt"] = tgt
        graph.add_edge(src, tgt, **attrs)
    graph.graph["hyperedges"] = data.get("hyperedges", [])
    return graph


def _prefix_graph_for_source(G: nx.Graph, manifest: dict[str, Any], source_id: str) -> nx.Graph:
    source = manifest["sources"][source_id]
    relabel = {node: f"{source_id}::{node}" for node in G.nodes}
    H = nx.relabel_nodes(G, relabel, copy=True)
    for node, data in H.nodes(data=True):
        local_id = node.split("::", 1)[1]
        data["workspace"] = manifest["name"]
        data["source_id"] = source_id
        data["source_kind"] = source.get("kind", "folder")
        data["source_path"] = source.get("path", "")
        data["local_id"] = local_id
        data.setdefault("repo", source_id)
    for _, _, data in H.edges(data=True):
        data["workspace"] = manifest["name"]
        data["source_id"] = source_id
        data["source_kind"] = source.get("kind", "folder")
        data["source_path"] = source.get("path", "")
        if data.get("_src") in relabel:
            data["_src"] = relabel[data["_src"]]
        if data.get("_tgt") in relabel:
            data["_tgt"] = relabel[data["_tgt"]]
    hyperedges: list[Any] = []
    for hyperedge in G.graph.get("hyperedges", []):
        if not isinstance(hyperedge, dict):
            hyperedges.append(hyperedge)
            continue
        prefixed = dict(hyperedge)
        if isinstance(prefixed.get("id"), str) and prefixed["id"]:
            prefixed["id"] = f"{source_id}::{prefixed['id']}"
        nodes = prefixed.get("nodes")
        if isinstance(nodes, list):
            prefixed["nodes"] = [relabel.get(node, node) for node in nodes]
        prefixed["workspace"] = manifest["name"]
        prefixed["source_id"] = source_id
        prefixed["source_kind"] = source.get("kind", "folder")
        prefixed["source_path"] = source.get("path", "")
        hyperedges.append(prefixed)
    H.graph["hyperedges"] = hyperedges
    return H


def compose_workspace_graph(
    manifest: dict[str, Any],
    source_graph_paths: dict[str, str | Path],
) -> nx.Graph:
    G = nx.Graph()
    G.graph["hyperedges"] = []
    workspace_node = f"workspace::{manifest['name']}"
    manifest_file = manifest.get("manifest_path", "")
    G.add_node(
        workspace_node,
        label=manifest["name"],
        file_type="concept",
        source_file=manifest_file,
        workspace=manifest["name"],
    )
    for source_id, source in manifest.get("sources", {}).items():
        source_node = f"source::{source_id}"
        G.add_node(
            source_node,
            label=source.get("label") or source_id,
            file_type="concept",
            source_file=manifest_file,
            workspace=manifest["name"],
            source_id=source_id,
            source_kind=source.get("kind", "folder"),
            source_path=source.get("path", ""),
        )
        G.add_edge(
            workspace_node,
            source_node,
            relation="contains_source",
            confidence="EXTRACTED",
            confidence_score=1.0,
            source_file=manifest_file,
            source_location=None,
            context="workspace",
            weight=1.0,
            _src=workspace_node,
            _tgt=source_node,
        )
        graph_path = Path(source_graph_paths[source_id])
        if not graph_path.exists():
            raise WorkspaceError(f"source graph not found for {source_id}: {graph_path}")
        source_graph = _prefix_graph_for_source(_load_graph(graph_path), manifest, source_id)
        hyperedges = list(G.graph.get("hyperedges", [])) + list(
            source_graph.graph.get("hyperedges", [])
        )
        G = nx.compose(G, source_graph)
        G.graph["hyperedges"] = hyperedges
    for relation in manifest.get("relations", []):
        src = f"source::{relation.get('source', '')}"
        tgt = f"source::{relation.get('target', '')}"
        if src not in G or tgt not in G:
            continue
        G.add_edge(
            src,
            tgt,
            relation=relation.get("relation", "relates_to"),
            confidence=relation.get("confidence", "EXTRACTED"),
            confidence_score=relation.get("confidence_score", 1.0),
            source_file=manifest_file,
            source_location=None,
            context="workspace",
            description=relation.get("description", ""),
            weight=relation.get("weight", 1.0),
            _src=src,
            _tgt=tgt,
        )
    return G


def _aggregate_detection(detections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"files": {}, "total_files": 0, "total_words": 0, "sources": {}}
    for source_id, detection in detections.items():
        merged["sources"][source_id] = detection
        merged["total_files"] += int(detection.get("total_files", 0) or 0)
        merged["total_words"] += int(detection.get("total_words", 0) or 0)
        for ftype, paths in detection.get("files", {}).items():
            merged["files"].setdefault(ftype, []).extend(paths)
    return merged


def collect_workspace_report_inputs(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    detections: dict[str, dict[str, Any]] = {}
    token_totals = {"input": 0, "output": 0}
    for source_id, source in manifest.get("sources", {}).items():
        detection = source.get("last_detection")
        if isinstance(detection, dict):
            detections[source_id] = detection
        tokens = source.get("last_tokens")
        if isinstance(tokens, dict):
            token_totals["input"] += int(tokens.get("input", 0) or 0)
            token_totals["output"] += int(tokens.get("output", 0) or 0)
    return detections, token_totals


def _extract_source_graph(
    source: dict[str, Any],
    output_root: Path,
    *,
    backend: str | None = None,
    model: str | None = None,
    google_workspace: bool | None = None,
    no_cluster: bool = False,
    max_workers: int | None = None,
) -> tuple[Path, dict[str, Any], dict[str, int]]:
    from graphify.detect import detect
    from graphify.extract import extract as ast_extract
    from graphify.build import build
    from graphify.cluster import cluster, score_all
    from graphify.export import to_json
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate

    source_path = Path(source["path"]).expanduser().resolve()
    if not source_path.exists():
        raise WorkspaceError(f"source path not found: {source_path}")
    graphify_out = output_root / "graphify-out"
    graphify_out.mkdir(parents=True, exist_ok=True)

    detection = detect(source_path, google_workspace=google_workspace)
    files_by_type = detection.get("files", {})
    code_files = [Path(p) for p in files_by_type.get("code", [])]
    semantic_files = [
        Path(p)
        for kind in ("document", "paper", "image")
        for p in files_by_type.get(kind, [])
    ]

    ast_result = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    if code_files:
        kwargs: dict[str, Any] = {"cache_root": source_path}
        if max_workers is not None:
            kwargs["max_workers"] = max_workers
        ast_result = ast_extract(code_files, **kwargs)

    sem_result = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    if semantic_files:
        from graphify.cache import check_semantic_cache, save_semantic_cache
        from graphify.llm import (
            extract_corpus_parallel,
        )

        chosen_backend = resolve_semantic_backend(backend)
        cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(
            [str(p) for p in semantic_files],
            root=source_path,
        )
        sem_result["nodes"].extend(cached_nodes)
        sem_result["edges"].extend(cached_edges)
        sem_result["hyperedges"].extend(cached_hyperedges)
        if uncached:
            fresh = extract_corpus_parallel(
                [Path(p) for p in uncached],
                backend=chosen_backend,
                model=model,
                root=source_path,
            )
            save_semantic_cache(
                fresh.get("nodes", []),
                fresh.get("edges", []),
                fresh.get("hyperedges", []),
                root=source_path,
            )
            sem_result["nodes"].extend(fresh.get("nodes", []))
            sem_result["edges"].extend(fresh.get("edges", []))
            sem_result["hyperedges"].extend(fresh.get("hyperedges", []))
            sem_result["input_tokens"] += fresh.get("input_tokens", 0)
            sem_result["output_tokens"] += fresh.get("output_tokens", 0)

    merged = {
        "nodes": list(ast_result.get("nodes", [])) + list(sem_result.get("nodes", [])),
        "edges": list(ast_result.get("edges", [])) + list(sem_result.get("edges", [])),
        "hyperedges": list(sem_result.get("hyperedges", [])),
        "input_tokens": ast_result.get("input_tokens", 0) + sem_result.get("input_tokens", 0),
        "output_tokens": ast_result.get("output_tokens", 0) + sem_result.get("output_tokens", 0),
    }
    if not merged["nodes"]:
        raise WorkspaceError(f"source produced no graph nodes: {source_path}")

    graph_path = graphify_out / "graph.json"
    tokens = {"input": merged["input_tokens"], "output": merged["output_tokens"]}
    graph = build([merged], dedup=True, root=source_path)
    if no_cluster:
        to_json(graph, {}, str(graph_path), force=True)
        return graph_path, detection, tokens

    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    gods = god_nodes(graph)
    surprises = surprising_connections(graph, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(graph, communities, labels)
    to_json(graph, communities, str(graph_path), force=True)
    (graphify_out / ".graphify_analysis.json").write_text(
        json.dumps(
            {
                "communities": {str(k): v for k, v in communities.items()},
                "cohesion": {str(k): v for k, v in cohesion.items()},
                "gods": gods,
                "surprises": surprises,
                "tokens": tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = generate(
        graph,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        tokens,
        str(source_path),
        suggested_questions=questions,
    )
    (graphify_out / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    return graph_path, detection, tokens


def build_workspace(
    name: str,
    *,
    source_id: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    google_workspace: bool | None = None,
    no_cluster: bool = False,
    max_workers: int | None = None,
) -> dict[str, Any]:
    from graphify.cluster import cluster, score_all
    from graphify.export import to_json
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate

    manifest = load_workspace(name)
    status = doctor_workspace(name)
    if status["missing_sources"]:
        missing = ", ".join(f"{m['id']} ({m['path']})" for m in status["missing_sources"])
        raise WorkspaceError(f"workspace has missing sources: {missing}")
    if status["non_directory_sources"]:
        non_directory = ", ".join(
            f"{m['id']} ({m['path']})" for m in status["non_directory_sources"]
        )
        raise WorkspaceError(f"workspace has non-directory sources: {non_directory}")
    if status.get("invalid_relations"):
        invalid = ", ".join(
            f"#{r['index']} {r['source']}->{r['target']} ({r['problem']})"
            for r in status["invalid_relations"]
        )
        raise WorkspaceError(f"workspace has invalid relations: {invalid}")

    selected = [source_id] if source_id else list(manifest.get("sources", {}).keys())
    if not selected:
        raise WorkspaceError(f"workspace has no sources: {name}")
    for sid in selected:
        if sid not in manifest["sources"]:
            raise WorkspaceError(f"source not found in workspace: {sid}")

    source_graphs: dict[str, Path] = {}
    for sid in manifest["sources"]:
        graph_path = _source_graph_path(name, sid)
        if sid in selected:
            graph_path, detection, tokens = _extract_source_graph(
                manifest["sources"][sid],
                graph_path.parent.parent,
                backend=backend,
                model=model,
                google_workspace=google_workspace,
                no_cluster=no_cluster,
                max_workers=max_workers,
            )
            manifest["sources"][sid]["last_seen"] = _now()
            manifest["sources"][sid]["last_detection"] = detection
            manifest["sources"][sid]["last_tokens"] = tokens
        elif not graph_path.exists():
            raise WorkspaceError(f"source graph missing for unchanged source: {sid}")
        source_graphs[sid] = graph_path

    manifest = save_workspace(manifest)
    detections, token_totals = collect_workspace_report_inputs(manifest)
    graph = compose_workspace_graph(manifest, source_graphs)
    graph_path = Path(manifest["graph_path"])
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    if no_cluster:
        to_json(graph, {}, str(graph_path), force=True)
        analysis_path = graph_path.parent / ".graphify_analysis.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "communities": {},
                    "cohesion": {},
                    "gods": [],
                    "surprises": [],
                    "tokens": token_totals,
                    "workspace": manifest["name"],
                    "no_cluster": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "workspace": manifest["name"],
            "graph_path": graph_path,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "sources": list(source_graphs),
        }

    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    gods = god_nodes(graph)
    surprises = surprising_connections(graph, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(graph, communities, labels)

    to_json(graph, communities, str(graph_path), force=True)
    analysis_path = graph_path.parent / ".graphify_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "communities": {str(k): v for k, v in communities.items()},
                "cohesion": {str(k): v for k, v in cohesion.items()},
                "gods": gods,
                "surprises": surprises,
                "tokens": token_totals,
                "workspace": manifest["name"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = generate(
        graph,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        _aggregate_detection(detections),
        token_totals,
        f"workspace:{manifest['name']}",
        suggested_questions=questions,
    )
    (graph_path.parent / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    return {
        "workspace": manifest["name"],
        "graph_path": graph_path,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "sources": list(source_graphs),
    }


def print_doctor(status: dict[str, Any]) -> int:
    has_errors = False
    for missing in status["missing_sources"]:
        has_errors = True
        print(
            f"missing source {missing['id']}: {missing['path']}",
            file=sys.stderr,
        )
    for source in status.get("non_directory_sources", []):
        has_errors = True
        print(
            f"source is not a directory {source['id']}: {source['path']}",
            file=sys.stderr,
        )
    if not status.get("graph_exists", False):
        has_errors = True
        print(f"workspace graph missing: {status['graph_path']}", file=sys.stderr)
    for source_graph in status.get("missing_source_graphs", []):
        has_errors = True
        print(
            f"source graph missing {source_graph['id']}: {source_graph['path']}",
            file=sys.stderr,
        )
    for relation in status.get("invalid_relations", []):
        has_errors = True
        print(
            f"invalid relation #{relation['index']} "
            f"{relation['source']}->{relation['target']}: {relation['problem']}",
            file=sys.stderr,
        )
    if has_errors:
        return 1
    print(f"workspace {status['workspace']} ok")
    print(f"graph: {status['graph_path']}")
    return 0
