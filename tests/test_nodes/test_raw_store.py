"""Raw (bronze) layer: write-once, content-addressed source texts."""

from __future__ import annotations

from pathlib import Path

from src.storage.raw_store import raw_uri_for, read_raw, store_raw
from src.utils.hashing import content_hash


async def test_text_round_trips_exactly(_isolated_raw_store: Path) -> None:
    text = "Line one\nLine two — with “quotes” and ünïcode\n"
    uri = await store_raw(content_hash(text), text)

    assert Path(uri).parent == _isolated_raw_store
    assert Path(uri).name == f"{content_hash(text)}.txt"
    assert await read_raw(uri) == text


async def test_same_content_is_stored_once(_isolated_raw_store: Path) -> None:
    text = "identical announcement"
    first = await store_raw(content_hash(text), text)
    second = await store_raw(content_hash(text), text)

    assert first == second
    assert len(list(_isolated_raw_store.iterdir())) == 1


async def test_existing_object_is_never_overwritten(_isolated_raw_store: Path) -> None:
    text = "original"
    key = content_hash(text)
    uri = await store_raw(key, text)
    # A (buggy) caller passing different text under the same key must not
    # change what is stored.
    await store_raw(key, "something else")
    assert await read_raw(uri) == "original"


async def test_an_edited_source_keeps_both_versions(_isolated_raw_store: Path) -> None:
    v1, v2 = "deal at $10B", "deal at $12B"
    uri1 = await store_raw(content_hash(v1), v1)
    uri2 = await store_raw(content_hash(v2), v2)

    assert uri1 != uri2
    assert await read_raw(uri1) == v1
    assert await read_raw(uri2) == v2


async def test_no_temporary_files_are_left_behind(_isolated_raw_store: Path) -> None:
    await store_raw(content_hash("x"), "x")
    assert [
        p.name for p in _isolated_raw_store.iterdir() if p.name.startswith(".")
    ] == []


def test_uri_is_derived_from_the_hash_only(_isolated_raw_store: Path) -> None:
    assert raw_uri_for("abc") == str(_isolated_raw_store / "abc.txt")


async def test_concurrent_writers_of_the_same_content(
    _isolated_raw_store: Path,
) -> None:
    import asyncio

    text = "same announcement, stored by many tasks at once"
    uris = await asyncio.gather(
        *(store_raw(content_hash(text), text) for _ in range(25))
    )

    assert len(set(uris)) == 1
    assert await read_raw(uris[0]) == text
    assert [p.name for p in _isolated_raw_store.iterdir()] == [
        f"{content_hash(text)}.txt"
    ]
