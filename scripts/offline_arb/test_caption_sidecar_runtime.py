#!/usr/bin/env python3
"""Node-local CPU self-test for the strict caption sidecar runtime reader.

The fixture is intentionally synthetic and tiny.  It exercises the same frozen
five-domain wire contract as production without reading any production catalog,
caption source, image, or sidecar artifact.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.offline_arb.caption_sidecar_contract import (  # noqa: E402
    ArrowIpcSemanticDigester,
    BUILDER_BUILD_ID,
    CAPTION_COLUMNS,
    CAPTION_SCHEMAS,
    CAPTION_SCHEMA_CONTRACT_SHA256,
    DOMAINS,
    FILE_ROWS,
    ROW_GROUP_ROWS,
    ROW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_DIGEST_ALGORITHM,
    header_self_hash,
    sidecar_schema as canonical_sidecar_schema,
)
from training.dataline_offline.caption_sidecar import (  # noqa: E402
    CaptionSidecarError,
    CaptionSidecarStore,
)
from training.dataline_offline.offline_dataset import (  # noqa: E402
    OfflineExactDataset,
    OfflineLoaderError,
    _PlannedItem,
)


ROWS_PER_DOMAIN = 513
CATALOG_HASH = hashlib.sha256(b"runtime-selftest-catalog").hexdigest()
CATALOG_HEADER_SHA256 = hashlib.sha256(
    b"runtime-selftest-catalog-header"
).hexdigest()
_ANIME_LIST_COLUMNS = {
    "tags_general",
    "tags_artist",
    "tags_character",
    "tags_copyright",
    "tags_meta",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _publish_header(root: Path, header: dict[str, Any]) -> None:
    header["header_sha256"] = header_self_hash(header)
    _write_json(root / "header.json", header)
    (root / "_SUCCESS").write_text(
        f"{header['header_sha256']}\n", encoding="ascii"
    )


def _caption_schema(domain: str) -> pa.Schema:
    return CAPTION_SCHEMAS[domain]


def _caption_value(domain: str, name: str, local_row: int) -> Any:
    if local_row == 1:
        return None
    if domain == "anime" and name in _ANIME_LIST_COLUMNS:
        return [f"{name}:{local_row}", "猫咪✨", "naïve café"]
    return f"{domain}/{name}/第{local_row}行/🌙"


def _semantic_digest(
    domain: str,
    row_base: int,
    row_count: int,
    schema: pa.Schema,
    tables: list[pa.Table],
) -> str:
    digester = ArrowIpcSemanticDigester(
        domain=domain,
        row_base=row_base,
        row_count=row_count,
        schema=schema,
    )
    for table in tables:
        digester.update(table)
    return digester.finish()


def _build_fixture(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    root.mkdir()
    domains_header: dict[str, Any] = {}
    catalog_files_header: dict[str, Any] = {}
    expected_domains: dict[str, dict[str, Any]] = {}
    expected_values: dict[str, dict[int, dict[str, Any]]] = {}
    row_base = 0

    for domain in DOMAINS:
        domain_root = root / domain
        domain_root.mkdir()
        caption_schema = _caption_schema(domain)
        sidecar_schema = canonical_sidecar_schema(
            caption_schema, CAPTION_COLUMNS[domain]
        )
        row_ids = list(range(row_base, row_base + ROWS_PER_DOMAIN))
        arrays: list[pa.Array] = [pa.array(row_ids, type=pa.int64())]
        values_by_column: dict[str, list[Any]] = {}
        for field in caption_schema:
            values = [
                _caption_value(domain, field.name, local_row)
                for local_row in range(ROWS_PER_DOMAIN)
            ]
            values_by_column[field.name] = values
            arrays.append(pa.array(values, type=field.type))
        table = pa.Table.from_arrays(arrays, schema=sidecar_schema)
        path = domain_root / "part-00000.parquet"
        pq.write_table(
            table,
            path,
            row_group_size=ROW_GROUP_ROWS,
            compression="zstd",
            use_dictionary=False,
        )
        parquet = pq.ParquetFile(path)
        _require(parquet.metadata.num_rows == ROWS_PER_DOMAIN, "fixture row drift")
        _require(
            parquet.num_row_groups == math.ceil(ROWS_PER_DOMAIN / ROW_GROUP_ROWS),
            "fixture row-group drift",
        )

        source_tables = [
            table.slice(offset, min(ROW_GROUP_ROWS, ROWS_PER_DOMAIN - offset))
            for offset in range(0, ROWS_PER_DOMAIN, ROW_GROUP_ROWS)
        ]
        decoded_tables = [
            parquet.read_row_group(
                row_group, columns=sidecar_schema.names, use_threads=False
            )
            for row_group in range(parquet.num_row_groups)
        ]
        source_digest = _semantic_digest(
            domain, row_base, ROWS_PER_DOMAIN, sidecar_schema, source_tables
        )
        decoded_digest = _semantic_digest(
            domain, row_base, ROWS_PER_DOMAIN, sidecar_schema, decoded_tables
        )
        _require(source_digest == decoded_digest, f"fixture digest drift: {domain}")

        file_entry = {
            "name": f"{domain}/part-00000.parquet",
            "row_start": row_base,
            "row_stop": row_base + ROWS_PER_DOMAIN,
            "rows": ROWS_PER_DOMAIN,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "row_groups": parquet.num_row_groups,
            "schema": str(parquet.schema_arrow),
        }
        domains_header[domain] = {
            "row_base": row_base,
            "row_count": ROWS_PER_DOMAIN,
            "columns": list(CAPTION_COLUMNS[domain]),
            "source_semantics_sha256": source_digest,
            "decoded_sidecar_semantics_sha256": decoded_digest,
            "files": [file_entry],
        }
        catalog_file_binding = {
            "name": f"catalog-{domain}.parquet",
            "row_base": row_base,
            "rows": ROWS_PER_DOMAIN,
            "bytes": 10_000 + len(domain),
            "sha256": hashlib.sha256(
                f"runtime-selftest-catalog-file:{domain}".encode("ascii")
            ).hexdigest(),
            "schema": f"runtime-selftest-catalog-schema/{domain}",
        }
        catalog_files_header[domain] = catalog_file_binding
        expected_domains[domain] = {
            "row_base": row_base,
            "row_count": ROWS_PER_DOMAIN,
            "columns": CAPTION_COLUMNS[domain],
            "catalog_file": {
                key: value
                for key, value in catalog_file_binding.items()
                if key != "row_base"
            }
            | {
                "compression": "zstd",
                "row_group_target_rows": ROW_GROUP_ROWS,
            },
        }
        expected_values[domain] = {
            row_base + local_row: {
                name: values_by_column[name][local_row]
                for name in CAPTION_COLUMNS[domain]
            }
            for local_row in range(ROWS_PER_DOMAIN)
        }
        row_base += ROWS_PER_DOMAIN

    header = {
        "schema_version": SCHEMA_VERSION,
        "builder_build_id": BUILDER_BUILD_ID,
        "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
        "catalog_hash": CATALOG_HASH,
        "catalog_header_sha256": CATALOG_HEADER_SHA256,
        "row_order": ROW_ORDER,
        "file_rows": FILE_ROWS,
        "row_group_rows": ROW_GROUP_ROWS,
        "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
        "domain_order": list(DOMAINS),
        "total_rows": row_base,
        "diagnostic_run": False,
        "diagnostic_rows_per_domain": None,
        "catalog_files": catalog_files_header,
        "domains": domains_header,
        "generated_at": "1970-01-01T00:00:00Z",
    }
    _publish_header(root, header)
    return expected_domains, expected_values


def _new_store(
    root: Path, expected_domains: dict[str, dict[str, Any]]
) -> CaptionSidecarStore:
    return CaptionSidecarStore(
        root,
        catalog_hash=CATALOG_HASH,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        expected_domains=expected_domains,
        workers=16,
    )


def _expect_caption_error(label: str, callback: Callable[[], Any]) -> None:
    try:
        callback()
    except CaptionSidecarError as exc:
        print(f"PASS {label}: {type(exc).__name__}: {exc}")
        return
    raise AssertionError(f"{label} did not fail closed")


def _expect_loader_error(label: str, callback: Callable[[], Any]) -> None:
    try:
        callback()
    except OfflineLoaderError as exc:
        print(f"PASS {label}: {type(exc).__name__}: {exc}")
        return
    raise AssertionError(f"{label} did not fail closed")


def _clone(root: Path, workspace: Path, name: str) -> Path:
    destination = workspace / name
    shutil.copytree(root, destination)
    return destination


def _read_header(root: Path) -> dict[str, Any]:
    value = json.loads((root / "header.json").read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "fixture header is not an object")
    return value


def _expect_header_mutation_failure(
    root: Path,
    workspace: Path,
    name: str,
    label: str,
    expected_domains: dict[str, dict[str, Any]],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    mutated_root = _clone(root, workspace, name)
    mutated_header = _read_header(mutated_root)
    mutate(mutated_header)
    _publish_header(mutated_root, mutated_header)
    _expect_caption_error(
        label, lambda: _new_store(mutated_root, expected_domains)
    )


def _update_file_declaration(
    root: Path,
    header: dict[str, Any],
    domain: str,
    *,
    update_schema: bool,
) -> None:
    path = root / domain / "part-00000.parquet"
    parquet = pq.ParquetFile(path)
    entry = header["domains"][domain]["files"][0]
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = _sha256_file(path)
    entry["row_groups"] = parquet.num_row_groups
    if update_schema:
        entry["schema"] = str(parquet.schema_arrow)


def _test_valid_reads(
    root: Path,
    expected_domains: dict[str, dict[str, Any]],
    expected_values: dict[str, dict[int, dict[str, Any]]],
) -> None:
    store = _new_store(root, expected_domains)
    queries: dict[str, list[int]] = {}
    anime_base = expected_domains["anime"]["row_base"]
    try:
        for domain in DOMAINS:
            base = expected_domains[domain]["row_base"]
            queries[domain] = [
                base,
                base + 1,
                base + 255,
                base + 256,
                base + 512,
                base + 256,
                base,
            ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [
                (domain, pool.submit(store.get_many, domain, query))
                for _round in range(2)
                for domain, query in queries.items()
            ]
            for domain, future in futures:
                rows = future.result()
                expected_ids = set(queries[domain])
                _require(set(rows) == expected_ids, f"result row set drift: {domain}")
                for row_id in expected_ids:
                    _require(
                        rows[row_id] == expected_values[domain][row_id],
                        f"typed row drift: {domain}/{row_id}",
                    )

        anime = store.get_many("anime", [anime_base, anime_base + 1])
        _require(
            anime[anime_base]["tags_general"]
            == ["tags_general:0", "猫咪✨", "naïve café"],
            "list/Unicode value did not round-trip",
        )
        _require(
            anime[anime_base + 1]["refined_description"] is None,
            "typed null did not round-trip",
        )
        _expect_caption_error(
            "out-of-range row_id",
            lambda: store.get_many("anime", [anime_base - 1]),
        )
        _expect_caption_error(
            "unknown domain", lambda: store.get_many("not-a-domain", [0])
        )
    finally:
        store.close()
    _expect_caption_error(
        "closed store", lambda: store.get_many("anime", [anime_base])
    )
    _expect_caption_error(
        "closed store empty read", lambda: store.get_many("anime", [])
    )
    print("PASS valid concurrent typed reads")


def _test_corruptions(
    root: Path,
    workspace: Path,
    expected_domains: dict[str, dict[str, Any]],
) -> None:
    corrupt_header_root = _clone(root, workspace, "corrupt-header")
    corrupt_header = _read_header(corrupt_header_root)
    corrupt_header["row_order"] = "corrupt"
    _write_json(corrupt_header_root / "header.json", corrupt_header)
    _expect_caption_error(
        "header self-hash corruption",
        lambda: _new_store(corrupt_header_root, expected_domains),
    )

    corrupt_success_root = _clone(root, workspace, "corrupt-success")
    (corrupt_success_root / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
    _expect_caption_error(
        "_SUCCESS corruption",
        lambda: _new_store(corrupt_success_root, expected_domains),
    )

    _expect_header_mutation_failure(
        root,
        workspace,
        "missing-top-key",
        "top-level missing key",
        expected_domains,
        lambda header: header.pop("total_rows"),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "extra-top-key",
        "top-level extra key",
        expected_domains,
        lambda header: header.__setitem__("unexpected_top_key", 1),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "missing-domain-key",
        "domain missing key",
        expected_domains,
        lambda header: header["domains"]["anime"].pop("columns"),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "extra-domain-key",
        "domain extra key",
        expected_domains,
        lambda header: header["domains"]["anime"].__setitem__(
            "unexpected_domain_key", 1
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "missing-file-key",
        "file missing key",
        expected_domains,
        lambda header: header["domains"]["anime"]["files"][0].pop(
            "row_groups"
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "extra-file-key",
        "file extra key",
        expected_domains,
        lambda header: header["domains"]["anime"]["files"][0].__setitem__(
            "unexpected_file_key", 1
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-total-rows",
        "total_rows corruption",
        expected_domains,
        lambda header: header.__setitem__(
            "total_rows", header["total_rows"] + 1
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-generated-at",
        "generated_at corruption",
        expected_domains,
        lambda header: header.__setitem__("generated_at", ""),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "missing-catalog-domain",
        "catalog_files missing domain",
        expected_domains,
        lambda header: header["catalog_files"].pop(DOMAINS[-1]),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "extra-catalog-domain",
        "catalog_files extra domain",
        expected_domains,
        lambda header: header["catalog_files"].__setitem__("unexpected", {}),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "missing-catalog-file-key",
        "catalog file missing key",
        expected_domains,
        lambda header: header["catalog_files"]["gpic"].pop("schema"),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "extra-catalog-file-key",
        "catalog file extra key",
        expected_domains,
        lambda header: header["catalog_files"]["gpic"].__setitem__(
            "unexpected_catalog_file_key", 1
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-catalog-file-binding",
        "catalog file binding corruption",
        expected_domains,
        lambda header: header["catalog_files"]["gpic"].__setitem__(
            "bytes", header["catalog_files"]["gpic"]["bytes"] + 1
        ),
    )

    wrong_catalog_root = _clone(root, workspace, "wrong-catalog")
    wrong_catalog = _read_header(wrong_catalog_root)
    wrong_catalog["catalog_hash"] = "0" * 64
    _publish_header(wrong_catalog_root, wrong_catalog)
    _expect_caption_error(
        "catalog binding corruption",
        lambda: _new_store(wrong_catalog_root, expected_domains),
    )

    wrong_algorithm_root = _clone(root, workspace, "wrong-semantic-algorithm")
    wrong_algorithm = _read_header(wrong_algorithm_root)
    wrong_algorithm["semantic_digest_algorithm"] = "unknown_algorithm_v0"
    _publish_header(wrong_algorithm_root, wrong_algorithm)
    _expect_caption_error(
        "semantic digest algorithm corruption",
        lambda: _new_store(wrong_algorithm_root, expected_domains),
    )

    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-schema-version",
        "schema_version contract corruption",
        expected_domains,
        lambda header: header.__setitem__(
            "schema_version", "offline_arb_caption_sidecar_v0"
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-row-order-contract",
        "row_order contract corruption",
        expected_domains,
        lambda header: header.__setitem__("row_order", "local_domain_row_id"),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-file-rows",
        "file_rows contract corruption",
        expected_domains,
        lambda header: header.__setitem__("file_rows", FILE_ROWS + 1),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-row-group-rows",
        "row_group_rows contract corruption",
        expected_domains,
        lambda header: header.__setitem__(
            "row_group_rows", ROW_GROUP_ROWS + 1
        ),
    )

    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-builder-build-id",
        "builder build-id corruption",
        expected_domains,
        lambda header: header.__setitem__(
            "builder_build_id", "offline_arb_caption_sidecar_builder_v2"
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-schema-contract-hash",
        "caption schema contract hash corruption",
        expected_domains,
        lambda header: header.__setitem__(
            "caption_schema_contract_sha256", "0" * 64
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "diagnostic-run",
        "diagnostic_run artifact",
        expected_domains,
        lambda header: header.__setitem__("diagnostic_run", True),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "diagnostic-row-limit",
        "diagnostic_rows_per_domain artifact",
        expected_domains,
        lambda header: header.__setitem__("diagnostic_rows_per_domain", 513),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "missing-diagnostic-row-limit",
        "missing diagnostic_rows_per_domain binding",
        expected_domains,
        lambda header: header.pop("diagnostic_rows_per_domain"),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-domain-order",
        "domain_order sequence corruption",
        expected_domains,
        lambda header: header.__setitem__(
            "domain_order", list(reversed(DOMAINS))
        ),
    )
    _expect_header_mutation_failure(
        root,
        workspace,
        "wrong-domain-order-set",
        "domain_order set corruption",
        expected_domains,
        lambda header: header.__setitem__("domain_order", list(DOMAINS[:-1])),
    )

    reversed_expected_domains = dict(reversed(tuple(expected_domains.items())))
    _expect_caption_error(
        "runtime expected domain order corruption",
        lambda: _new_store(root, reversed_expected_domains),
    )
    missing_expected_domain = dict(expected_domains)
    missing_expected_domain.pop(DOMAINS[-1])
    _expect_caption_error(
        "runtime expected domain set corruption",
        lambda: _new_store(root, missing_expected_domain),
    )
    wrong_expected_columns = dict(expected_domains)
    wrong_expected_columns["gpic"] = {
        **wrong_expected_columns["gpic"],
        "columns": ("caption_drift",),
    }
    _expect_caption_error(
        "runtime expected canonical schema corruption",
        lambda: _new_store(root, wrong_expected_columns),
    )
    wrong_expected_catalog = dict(expected_domains)
    wrong_expected_catalog_file = dict(
        wrong_expected_catalog["gpic"]["catalog_file"]
    )
    wrong_expected_catalog_file["bytes"] += 1
    wrong_expected_catalog["gpic"] = {
        **wrong_expected_catalog["gpic"],
        "catalog_file": wrong_expected_catalog_file,
    }
    _expect_caption_error(
        "runtime catalog header binding corruption",
        lambda: _new_store(root, wrong_expected_catalog),
    )

    for domain in DOMAINS:
        _expect_header_mutation_failure(
            root,
            workspace,
            f"wrong-declared-schema-{domain}",
            f"declared canonical schema corruption for {domain}",
            expected_domains,
            lambda header, domain=domain: header["domains"][domain]["files"][
                0
            ].__setitem__("schema", "row_id: int64 not null\ncaption: int64"),
        )

    missing_header_domain_root = _clone(
        root, workspace, "missing-header-domain"
    )
    missing_header_domain = _read_header(missing_header_domain_root)
    missing_header_domain["domains"].pop(DOMAINS[-1])
    _publish_header(missing_header_domain_root, missing_header_domain)
    _expect_caption_error(
        "header domain set corruption",
        lambda: _new_store(missing_header_domain_root, expected_domains),
    )

    digest_mismatch_root = _clone(root, workspace, "semantic-digest-mismatch")
    digest_mismatch = _read_header(digest_mismatch_root)
    digest_mismatch["domains"]["monet_real"][
        "decoded_sidecar_semantics_sha256"
    ] = "0" * 64
    _publish_header(digest_mismatch_root, digest_mismatch)
    _expect_caption_error(
        "source/decoded semantic digest mismatch",
        lambda: _new_store(digest_mismatch_root, expected_domains),
    )

    malformed_hash_root = _clone(root, workspace, "malformed-file-hash")
    malformed_hash = _read_header(malformed_hash_root)
    malformed_hash["domains"]["anime"]["files"][0]["sha256"] = "not-a-sha256"
    _publish_header(malformed_hash_root, malformed_hash)
    _expect_caption_error(
        "malformed file hash declaration",
        lambda: _new_store(malformed_hash_root, expected_domains),
    )

    missing_root = _clone(root, workspace, "missing-file")
    (missing_root / "textatlas" / "part-00000.parquet").unlink()
    _expect_caption_error(
        "missing sidecar file", lambda: _new_store(missing_root, expected_domains)
    )

    row_id_root = _clone(root, workspace, "corrupt-row-id")
    row_id_header = _read_header(row_id_root)
    row_id_path = row_id_root / "anime" / "part-00000.parquet"
    row_id_table = pq.read_table(row_id_path, use_threads=False)
    row_ids = row_id_table.column("row_id").to_pylist()
    row_ids[5] = row_ids[5] + 99
    row_id_arrays = [
        pa.array(row_ids, type=pa.int64()),
        *[
            row_id_table.column(name).combine_chunks()
            for name in row_id_table.schema.names[1:]
        ],
    ]
    row_id_table = pa.Table.from_arrays(row_id_arrays, schema=row_id_table.schema)
    pq.write_table(
        row_id_table,
        row_id_path,
        row_group_size=ROW_GROUP_ROWS,
        compression="zstd",
        use_dictionary=False,
    )
    _update_file_declaration(
        row_id_root, row_id_header, "anime", update_schema=True
    )
    _publish_header(row_id_root, row_id_header)
    row_id_store = _new_store(row_id_root, expected_domains)
    try:
        anime_base = expected_domains["anime"]["row_base"]
        _expect_caption_error(
            "physical row_id corruption",
            lambda: row_id_store.get_many("anime", [anime_base + 5]),
        )
    finally:
        row_id_store.close()

    schema_root = _clone(root, workspace, "corrupt-schema")
    schema_header = _read_header(schema_root)
    schema_path = schema_root / "gpic" / "part-00000.parquet"
    gpic_base = expected_domains["gpic"]["row_base"]
    bad_schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("caption", pa.int64(), nullable=True),
        ]
    )
    bad_table = pa.Table.from_arrays(
        [
            pa.array(
                range(gpic_base, gpic_base + ROWS_PER_DOMAIN), type=pa.int64()
            ),
            pa.array(range(ROWS_PER_DOMAIN), type=pa.int64()),
        ],
        schema=bad_schema,
    )
    pq.write_table(
        bad_table,
        schema_path,
        row_group_size=ROW_GROUP_ROWS,
        compression="zstd",
        use_dictionary=False,
    )
    # Update physical identity declarations, but deliberately retain the old
    # declared schema so the runtime must reject the live parquet footer.
    _update_file_declaration(
        schema_root, schema_header, "gpic", update_schema=False
    )
    _publish_header(schema_root, schema_header)
    schema_store = _new_store(schema_root, expected_domains)
    try:
        _expect_caption_error(
            "physical schema corruption",
            lambda: schema_store.get_many("gpic", [gpic_base]),
        )
    finally:
        schema_store.close()

    metadata_root = _clone(root, workspace, "corrupt-schema-metadata")
    metadata_header = _read_header(metadata_root)
    metadata_path = metadata_root / "synthetic" / "part-00000.parquet"
    metadata_table = pq.read_table(metadata_path, use_threads=False)
    metadata_table = metadata_table.replace_schema_metadata(
        {b"runtime-selftest-unexpected": b"metadata"}
    )
    pq.write_table(
        metadata_table,
        metadata_path,
        row_group_size=ROW_GROUP_ROWS,
        compression="zstd",
        use_dictionary=False,
    )
    _update_file_declaration(
        metadata_root, metadata_header, "synthetic", update_schema=False
    )
    _publish_header(metadata_root, metadata_header)
    metadata_store = _new_store(metadata_root, expected_domains)
    try:
        synthetic_base = expected_domains["synthetic"]["row_base"]
        _expect_caption_error(
            "physical schema metadata corruption",
            lambda: metadata_store.get_many("synthetic", [synthetic_base]),
        )
    finally:
        metadata_store.close()


def _with_module_open(module: Any, replacement: Callable[..., Any], callback: Callable[[], Any]) -> Any:
    sentinel = object()
    previous = getattr(module, "open", sentinel)
    setattr(module, "open", replacement)
    try:
        return callback()
    finally:
        if previous is sentinel:
            delattr(module, "open")
        else:
            setattr(module, "open", previous)


def _with_module_stat_counts(
    module: Any, callback: Callable[[], Any]
) -> tuple[Any, dict[str, int]]:
    real_fstat = module.os.fstat
    real_stat = module.os.stat
    counts = {"fstat_calls": 0, "path_stat_calls": 0}

    def counted_fstat(*args: Any, **kwargs: Any) -> Any:
        counts["fstat_calls"] += 1
        return real_fstat(*args, **kwargs)

    def counted_stat(*args: Any, **kwargs: Any) -> Any:
        counts["path_stat_calls"] += 1
        return real_stat(*args, **kwargs)

    module.os.fstat = counted_fstat
    module.os.stat = counted_stat
    try:
        return callback(), counts
    finally:
        module.os.fstat = real_fstat
        module.os.stat = real_stat


class _ReaderProxy:
    def __init__(self, handle: Any):
        self.handle = handle

    def __enter__(self) -> "_ReaderProxy":
        self.handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.handle.__exit__(*args)

    def fileno(self) -> int:
        return self.handle.fileno()


class _EarlyEofReader(_ReaderProxy):
    def __init__(self, handle: Any):
        super().__init__(handle)
        self.first = True

    def read(self, _size: int) -> bytes:
        if self.first:
            self.first = False
            return self.handle.read(1)
        return b""


class _MutatingReader(_ReaderProxy):
    def __init__(self, handle: Any, path: Path):
        super().__init__(handle)
        self.path = path
        self.first = True

    def read(self, size: int) -> bytes:
        block = self.handle.read(size)
        if self.first:
            self.first = False
            metadata = self.path.stat()
            os.utime(
                self.path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
            )
        return block


class _NamespaceSwapReader(_ReaderProxy):
    def __init__(self, handle: Any, visible_path: Path, replacement_link: Path):
        super().__init__(handle)
        self.visible_path = visible_path
        self.replacement_link = replacement_link
        self.first = True

    def read(self, size: int) -> bytes:
        block = self.handle.read(size)
        if self.first:
            self.first = False
            os.replace(self.replacement_link, self.visible_path)
        return block


def _test_gpic_memory_reader(workspace: Path) -> None:
    import training.dataline_offline.offline_dataset as offline_module

    source_a = workspace / "gpic-a.img"
    source_b = workspace / "gpic-b.img"
    source_link = workspace / "gpic-link.img"
    empty = workspace / "gpic-empty.img"
    source_a.write_bytes(b"direct-source-a\x00\xff")
    source_b.write_bytes(("GPIC/内存/🌙".encode("utf-8") + b"\x00") * 17)
    source_link.symlink_to(source_a)
    empty.write_bytes(b"")

    dataset = object.__new__(OfflineExactDataset)
    dataset.rank = 0
    # Serial execution makes the Python-level syscall counter deterministic;
    # production concurrency is orthogonal to this reader contract test.
    dataset._gpic_workers = 1
    dataset._pipeline_active = False
    dataset._caption_sidecar = None
    dataset._threads = []

    def pair(step: int, slot: int, row_id: int, path: Path) -> tuple[Any, dict[str, str]]:
        return (
            _PlannedItem(step, slot, row_id, "gpic", 2, 0, 0),
            {"image_path": str(path)},
        )

    pairs = [pair(3, 0, 100, source_a), pair(3, 1, 101, source_b)]
    expected = {(3, 0): source_a.read_bytes(), (3, 1): source_b.read_bytes()}
    for mode, expected_path_stats_per_file in (("strict_path", 2), ("fd_stable", 0)):
        dataset._gpic_source_consistency_mode = mode
        plain, plain_counts = _with_module_stat_counts(
            offline_module,
            lambda: dataset._gpic_payloads_in_memory(pairs, telemetry=False),
        )
        instrumented_result, instrumented_counts = _with_module_stat_counts(
            offline_module,
            lambda: dataset._gpic_payloads_in_memory(pairs, telemetry=True),
        )
        instrumented, detail = instrumented_result
        _require(plain == expected, f"{mode} plain GPIC bytes drift")
        _require(instrumented == expected, f"{mode} instrumented GPIC bytes drift")
        _require(plain == instrumented, f"{mode} plain/instrumented bytes differ")
        expected_counts = {
            "fstat_calls": 2 * len(pairs),
            "path_stat_calls": expected_path_stats_per_file * len(pairs),
        }
        _require(
            plain_counts == expected_counts,
            f"{mode} plain syscall counts drift: {plain_counts} != {expected_counts}",
        )
        _require(
            instrumented_counts == expected_counts,
            f"{mode} instrumented syscall counts drift: "
            f"{instrumented_counts} != {expected_counts}",
        )
        _require(
            detail.get("gpic_source_consistency_mode") == mode,
            f"{mode} telemetry mode identity drift",
        )
        _require(
            detail.get("fstat_calls") == expected_counts["fstat_calls"]
            and detail.get("path_stat_calls")
            == expected_counts["path_stat_calls"],
            f"{mode} telemetry syscall aggregate drift: {detail}",
        )
        _require(
            detail.get("files") == len(pairs),
            f"{mode} telemetry file aggregate drift",
        )

        link_pair = [pair(4, 0, 102, source_link)]
        link_plain, link_plain_counts = _with_module_stat_counts(
            offline_module,
            lambda: dataset._gpic_payloads_in_memory(link_pair, telemetry=False),
        )
        link_instrumented_result, link_instrumented_counts = (
            _with_module_stat_counts(
                offline_module,
                lambda: dataset._gpic_payloads_in_memory(
                    link_pair, telemetry=True
                ),
            )
        )
        link_instrumented, link_detail = link_instrumented_result
        expected_link = {(4, 0): expected[(3, 0)]}
        expected_link_counts = {
            "fstat_calls": 2,
            "path_stat_calls": expected_path_stats_per_file,
        }
        _require(
            link_plain == expected_link and link_instrumented == expected_link,
            f"{mode} ordinary-open symlink semantics drift",
        )
        _require(
            link_plain_counts == expected_link_counts
            and link_instrumented_counts == expected_link_counts,
            f"{mode} symlink syscall counts drift",
        )
        _require(
            link_detail.get("fstat_calls") == 2
            and link_detail.get("path_stat_calls")
            == expected_path_stats_per_file,
            f"{mode} symlink telemetry syscall counts drift",
        )

    _require(source_a.is_file() and source_b.is_file(), "GPIC source was removed")

    for mode in ("strict_path", "fd_stable"):
        dataset._gpic_source_consistency_mode = mode
        for telemetry in (False, True):
            _expect_loader_error(
                f"{mode} empty GPIC source telemetry={telemetry}",
                lambda telemetry=telemetry: dataset._gpic_payloads_in_memory(
                    [pair(5, 0, 103, empty)], telemetry=telemetry
                ),
            )

    real_open = open

    def short_open(path: str, *args: Any, **kwargs: Any) -> _EarlyEofReader:
        return _EarlyEofReader(real_open(path, *args, **kwargs))

    def mutating_open(path: str, *args: Any, **kwargs: Any) -> _MutatingReader:
        return _MutatingReader(real_open(path, *args, **kwargs), Path(path))

    for mode in ("strict_path", "fd_stable"):
        dataset._gpic_source_consistency_mode = mode
        for telemetry in (False, True):
            _expect_loader_error(
                f"{mode} short GPIC source read telemetry={telemetry}",
                lambda telemetry=telemetry: _with_module_open(
                    offline_module,
                    short_open,
                    lambda: dataset._gpic_payloads_in_memory(
                        [pair(6, 0, 104, source_a)], telemetry=telemetry
                    ),
                ),
            )
            _expect_loader_error(
                f"{mode} in-place GPIC change telemetry={telemetry}",
                lambda telemetry=telemetry: _with_module_open(
                    offline_module,
                    mutating_open,
                    lambda: dataset._gpic_payloads_in_memory(
                        [pair(7, 0, 105, source_b)], telemetry=telemetry
                    ),
                ),
            )

    old_namespace_bytes = b"namespace-old-inode\x00"
    new_namespace_bytes = b"namespace-new-inode\xff"
    for mode in ("strict_path", "fd_stable"):
        dataset._gpic_source_consistency_mode = mode
        for telemetry in (False, True):
            suffix = f"{mode}-{'instrumented' if telemetry else 'plain'}"
            old_target = workspace / f"gpic-namespace-old-{suffix}.img"
            new_target = workspace / f"gpic-namespace-new-{suffix}.img"
            visible_link = workspace / f"gpic-namespace-visible-{suffix}.img"
            replacement_link = workspace / f"gpic-namespace-replacement-{suffix}.img"
            old_target.write_bytes(old_namespace_bytes)
            new_target.write_bytes(new_namespace_bytes)
            visible_link.symlink_to(old_target)
            replacement_link.symlink_to(new_target)

            def namespace_open(
                path: str, *args: Any, **kwargs: Any
            ) -> _NamespaceSwapReader:
                return _NamespaceSwapReader(
                    real_open(path, *args, **kwargs),
                    visible_link,
                    replacement_link,
                )

            def read_namespace() -> Any:
                return _with_module_open(
                    offline_module,
                    namespace_open,
                    lambda: dataset._gpic_payloads_in_memory(
                        [pair(8, 0, 106, visible_link)], telemetry=telemetry
                    ),
                )

            if mode == "strict_path":
                _expect_loader_error(
                    f"strict namespace swap telemetry={telemetry}",
                    read_namespace,
                )
            else:
                namespace_result = read_namespace()
                if telemetry:
                    namespace_payloads, namespace_detail = namespace_result
                    _require(
                        namespace_detail.get("fstat_calls") == 2
                        and namespace_detail.get("path_stat_calls") == 0,
                        "fd_stable namespace telemetry syscall counts drift",
                    )
                else:
                    namespace_payloads = namespace_result
                _require(
                    namespace_payloads[(8, 0)] == old_namespace_bytes,
                    "fd_stable did not retain bytes from the opened old inode",
                )
            _require(
                visible_link.read_bytes() == new_namespace_bytes,
                f"{mode} namespace swap did not publish the replacement link",
            )

    _require(
        offline_module._validated_gpic_source_consistency_mode("strict_path")
        == "strict_path"
        and offline_module._validated_gpic_source_consistency_mode("fd_stable")
        == "fd_stable",
        "GPIC mode validator changed accepted values",
    )
    for invalid_mode in ("", "STRICT_PATH", "fd", True, None):
        _expect_loader_error(
            f"invalid GPIC consistency mode {invalid_mode!r}",
            lambda invalid_mode=invalid_mode: (
                offline_module._validated_gpic_source_consistency_mode(
                    invalid_mode
                )
            ),
        )
    resolve_threads = offline_module._resolved_gpic_threads_per_node
    _require(resolve_threads(None, None, 8) == 32, "GPIC legacy default drift")
    _require(resolve_threads(16, None, 8) == 16, "GPIC config binding drift")
    _require(resolve_threads(None, 16, 8) == 16, "GPIC env fallback drift")
    _require(resolve_threads(16, 16, 8) == 16, "GPIC equal config/env drift")
    for label, configured, environment, local_world in (
        ("GPIC config/env mismatch", 16, 32, 8),
        ("GPIC non-divisible config", 12, None, 8),
        ("GPIC below local world", 4, None, 8),
        ("GPIC boolean config", True, None, 8),
        ("GPIC zero config", 0, None, 8),
    ):
        _expect_loader_error(
            label,
            lambda configured=configured, environment=environment, local_world=local_world: (
                resolve_threads(configured, environment, local_world)
            ),
        )
    print("PASS GPIC strict_path/fd_stable direct-memory payload contract")


def main() -> int:
    local_base = Path(
        os.environ.get(
            "SELFTEST_LOCAL_ROOT",
            "/local/evan/offline_arb_caption_sidecar_runtime_selftest",
        )
    )
    local_base.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=local_base))
    try:
        fixture_root = workspace / "valid-sidecar"
        expected_domains, expected_values = _build_fixture(fixture_root)
        _test_valid_reads(fixture_root, expected_domains, expected_values)
        _test_corruptions(fixture_root, workspace, expected_domains)
        _test_gpic_memory_reader(workspace)
        print(
            json.dumps(
                {
                    "schema_version": "offline_arb_caption_sidecar_runtime_selftest_v1",
                    "status": "PASS",
                    "domains": list(DOMAINS),
                    "rows_per_domain": ROWS_PER_DOMAIN,
                    "row_group_rows": ROW_GROUP_ROWS,
                    "sidecar_workers": 16,
                },
                sort_keys=True,
            )
        )
    except BaseException:
        print(f"SELFTEST_FAILED_WORKSPACE={workspace}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
