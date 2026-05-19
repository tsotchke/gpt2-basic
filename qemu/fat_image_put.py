#!/usr/bin/env python3
"""Minimal FAT12/FAT16 image writer for the GPT2-BASIC QEMU images."""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path


END12 = 0xFF8
END16 = 0xFFF8


@dataclass
class DirEntry:
    name: str
    attr: int
    cluster: int
    size: int
    offset: int


class FATImage:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fp = path.open("r+b")
        self.base = self._find_base()
        self._read_bpb()

    def close(self) -> None:
        self.fp.close()

    def _read_at(self, offset: int, size: int) -> bytes:
        self.fp.seek(offset)
        data = self.fp.read(size)
        if len(data) != size:
            raise IOError(f"short read at {offset}")
        return data

    def _write_at(self, offset: int, data: bytes) -> None:
        self.fp.seek(offset)
        self.fp.write(data)

    def _find_base(self) -> int:
        sector = self._read_at(0, 512)
        bps = struct.unpack_from("<H", sector, 11)[0]
        if bps in {512, 1024, 2048, 4096} and sector[510:512] == b"\x55\xaa":
            return 0

        part_type = sector[450]
        lba_start = struct.unpack_from("<I", sector, 454)[0]
        if sector[510:512] == b"\x55\xaa" and part_type != 0 and lba_start > 0:
            return lba_start * 512

        raise ValueError(f"{self.path} does not look like a FAT image")

    def _read_bpb(self) -> None:
        b = self._read_at(self.base, 64)
        self.bps = struct.unpack_from("<H", b, 11)[0]
        self.spc = b[13]
        self.reserved = struct.unpack_from("<H", b, 14)[0]
        self.num_fats = b[16]
        self.root_entries = struct.unpack_from("<H", b, 17)[0]
        total16 = struct.unpack_from("<H", b, 19)[0]
        total32 = struct.unpack_from("<I", b, 32)[0]
        self.total_sectors = total16 if total16 else total32
        self.fat_sectors = struct.unpack_from("<H", b, 22)[0]
        self.root_sectors = ((self.root_entries * 32) + self.bps - 1) // self.bps
        self.fat_start = self.base + self.reserved * self.bps
        self.root_start = self.base + (self.reserved + self.num_fats * self.fat_sectors) * self.bps
        self.data_start = self.root_start + self.root_sectors * self.bps
        self.cluster_size = self.bps * self.spc
        data_sectors = self.total_sectors - (self.reserved + self.num_fats * self.fat_sectors + self.root_sectors)
        self.cluster_count = data_sectors // self.spc
        self.fat_bits = 12 if self.cluster_count < 4085 else 16
        self.end_marker = END12 if self.fat_bits == 12 else END16

    def cluster_offset(self, cluster: int) -> int:
        return self.data_start + (cluster - 2) * self.cluster_size

    def fat_get(self, cluster: int) -> int:
        if self.fat_bits == 16:
            return struct.unpack_from("<H", self._read_at(self.fat_start + cluster * 2, 2))[0]

        pos = self.fat_start + (cluster * 3) // 2
        pair = struct.unpack_from("<H", self._read_at(pos, 2))[0]
        if cluster & 1:
            return pair >> 4
        return pair & 0x0FFF

    def fat_set_one(self, fat_index: int, cluster: int, value: int) -> None:
        fat_start = self.fat_start + fat_index * self.fat_sectors * self.bps
        if self.fat_bits == 16:
            self._write_at(fat_start + cluster * 2, struct.pack("<H", value & 0xFFFF))
            return

        pos = fat_start + (cluster * 3) // 2
        pair = struct.unpack_from("<H", self._read_at(pos, 2))[0]
        if cluster & 1:
            pair = (pair & 0x000F) | ((value & 0x0FFF) << 4)
        else:
            pair = (pair & 0xF000) | (value & 0x0FFF)
        self._write_at(pos, struct.pack("<H", pair))

    def fat_set(self, cluster: int, value: int) -> None:
        for fat_index in range(self.num_fats):
            self.fat_set_one(fat_index, cluster, value)

    def is_eoc(self, value: int) -> bool:
        return value >= (END12 if self.fat_bits == 12 else END16)

    def cluster_chain(self, start: int) -> list[int]:
        if start < 2:
            return []
        chain: list[int] = []
        seen: set[int] = set()
        cluster = start
        while cluster >= 2 and cluster not in seen:
            seen.add(cluster)
            chain.append(cluster)
            value = self.fat_get(cluster)
            if value == 0 or self.is_eoc(value):
                break
            cluster = value
        return chain

    def free_chain(self, start: int) -> None:
        for cluster in self.cluster_chain(start):
            self.fat_set(cluster, 0)
            self._write_at(self.cluster_offset(cluster), b"\x00" * self.cluster_size)

    def alloc_clusters(self, count: int) -> list[int]:
        if count <= 0:
            return []
        clusters: list[int] = []
        for cluster in range(2, self.cluster_count + 2):
            if self.fat_get(cluster) == 0:
                clusters.append(cluster)
                if len(clusters) == count:
                    break
        if len(clusters) != count:
            raise IOError("FAT image is full")
        for idx, cluster in enumerate(clusters):
            next_value = clusters[idx + 1] if idx + 1 < len(clusters) else self.end_marker
            self.fat_set(cluster, next_value)
            self._write_at(self.cluster_offset(cluster), b"\x00" * self.cluster_size)
        return clusters

    @staticmethod
    def short_name(name: str) -> bytes:
        if name == ".":
            return b".          "
        if name == "..":
            return b"..         "

        upper = name.upper()
        base, dot, ext = upper.partition(".")
        if not base or len(base) > 8 or len(ext) > 3 or "." in ext:
            raise ValueError(f"not an 8.3 name: {name}")
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$~!#%&-{}()@'`"
        if any(ch not in allowed for ch in base + ext):
            raise ValueError(f"unsupported FAT name: {name}")
        return base.ljust(8).encode("ascii") + ext.ljust(3).encode("ascii")

    @staticmethod
    def display_name(raw: bytes) -> str:
        base = raw[:8].decode("ascii", errors="ignore").rstrip()
        ext = raw[8:11].decode("ascii", errors="ignore").rstrip()
        return f"{base}.{ext}" if ext else base

    def dir_region(self, cluster: int | None) -> tuple[list[int], bool]:
        if cluster is None:
            return [self.root_start + i * 32 for i in range(self.root_entries)], True
        offsets: list[int] = []
        for clus in self.cluster_chain(cluster):
            start = self.cluster_offset(clus)
            offsets.extend(start + i * 32 for i in range(self.cluster_size // 32))
        return offsets, False

    def list_dir(self, cluster: int | None) -> list[DirEntry]:
        entries: list[DirEntry] = []
        offsets, _is_root = self.dir_region(cluster)
        for offset in offsets:
            entry = self._read_at(offset, 32)
            first = entry[0]
            if first == 0x00:
                continue
            if first == 0xE5:
                continue
            attr = entry[11]
            if attr == 0x0F:
                continue
            raw_name = entry[:11]
            start_cluster = struct.unpack_from("<H", entry, 26)[0]
            size = struct.unpack_from("<I", entry, 28)[0]
            entries.append(DirEntry(self.display_name(raw_name), attr, start_cluster, size, offset))
        return entries

    def find_entry(self, cluster: int | None, name: str) -> DirEntry | None:
        raw = self.short_name(name)
        for offset, _ in [(o, None) for o in self.dir_region(cluster)[0]]:
            entry = self._read_at(offset, 32)
            if entry[0] in {0x00, 0xE5} or entry[11] == 0x0F:
                continue
            if entry[:11] == raw:
                return DirEntry(self.display_name(entry[:11]), entry[11], struct.unpack_from("<H", entry, 26)[0], struct.unpack_from("<I", entry, 28)[0], offset)
        return None

    def free_slot(self, cluster: int | None) -> int:
        offsets, is_root = self.dir_region(cluster)
        for offset in offsets:
            first = self._read_at(offset, 1)[0]
            if first in {0x00, 0xE5}:
                return offset
        if is_root:
            raise IOError("root directory is full")
        raise IOError("subdirectory is full")

    def write_entry(self, dir_cluster: int | None, name: str, attr: int, start_cluster: int, size: int) -> None:
        slot = self.free_slot(dir_cluster)
        entry = bytearray(32)
        entry[:11] = self.short_name(name)
        entry[11] = attr
        struct.pack_into("<H", entry, 26, start_cluster)
        struct.pack_into("<I", entry, 28, size)
        self._write_at(slot, bytes(entry))

    def mark_deleted(self, entry: DirEntry) -> None:
        self._write_at(entry.offset, b"\xE5")

    def path_parts(self, path: str) -> list[str]:
        return [part for part in path.replace("\\", "/").split("/") if part]

    def resolve_parent(self, path: str, create: bool = False) -> tuple[int | None, str]:
        parts = self.path_parts(path)
        if not parts:
            raise ValueError("empty path")
        cluster: int | None = None
        for part in parts[:-1]:
            entry = self.find_entry(cluster, part)
            if entry is None:
                if not create:
                    raise FileNotFoundError(part)
                self.mkdir_at(cluster, part)
                entry = self.find_entry(cluster, part)
                if entry is None:
                    raise IOError(f"failed to create {part}")
            if (entry.attr & 0x10) == 0:
                raise NotADirectoryError(part)
            cluster = entry.cluster
        return cluster, parts[-1]

    def remove(self, path: str) -> None:
        parent, name = self.resolve_parent(path)
        entry = self.find_entry(parent, name)
        if entry is None:
            return
        if entry.attr & 0x10:
            for child in list(self.list_dir(entry.cluster)):
                if child.name in {".", ".."}:
                    continue
                self.remove(path.rstrip("/\\") + "/" + child.name)
        if entry.cluster:
            self.free_chain(entry.cluster)
        self.mark_deleted(entry)

    def mkdir_at(self, parent_cluster: int | None, name: str) -> None:
        existing = self.find_entry(parent_cluster, name)
        if existing is not None:
            if existing.attr & 0x10:
                return
            raise FileExistsError(name)
        cluster = self.alloc_clusters(1)[0]
        self.write_entry(parent_cluster, name, 0x10, cluster, 0)
        self.write_entry(cluster, ".", 0x10, cluster, 0)
        self.write_entry(cluster, "..", 0x10, parent_cluster or 0, 0)

    def mkdir(self, path: str) -> None:
        parent, name = self.resolve_parent(path, create=True)
        self.mkdir_at(parent, name)

    def write_file(self, path: str, data: bytes) -> None:
        parent, name = self.resolve_parent(path, create=True)
        existing = self.find_entry(parent, name)
        if existing is not None:
            self.remove(path)
            parent, name = self.resolve_parent(path, create=True)

        count = (len(data) + self.cluster_size - 1) // self.cluster_size
        clusters = self.alloc_clusters(count)
        for idx, cluster in enumerate(clusters):
            chunk = data[idx * self.cluster_size : (idx + 1) * self.cluster_size]
            self._write_at(self.cluster_offset(cluster), chunk.ljust(self.cluster_size, b"\x00"))
        self.write_entry(parent, name, 0x20, clusters[0] if clusters else 0, len(data))

    def read_file(self, path: str) -> bytes:
        parent, name = self.resolve_parent(path)
        entry = self.find_entry(parent, name)
        if entry is None:
            raise FileNotFoundError(path)
        if entry.attr & 0x10:
            raise IsADirectoryError(path)

        chunks: list[bytes] = []
        for cluster in self.cluster_chain(entry.cluster):
            chunks.append(self._read_at(self.cluster_offset(cluster), self.cluster_size))
        return b"".join(chunks)[: entry.size]


def put_tree(image: FATImage, source: Path, dest: str) -> None:
    image.remove(dest)
    image.mkdir(dest)
    for root, dirs, files in os.walk(source):
        rel = Path(root).relative_to(source)
        dest_root = dest if str(rel) == "." else dest.rstrip("/\\") + "/" + str(rel).replace(os.sep, "/")
        for dirname in sorted(dirs):
            image.mkdir(dest_root + "/" + dirname)
        for filename in sorted(files):
            path = Path(root) / filename
            image.write_file(dest_root + "/" + filename, path.read_bytes())


def normalize_dos_text(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def self_test() -> None:
    source_image = Path(__file__).resolve().parent / "boot-test.img"
    if not source_image.exists():
        raise FileNotFoundError(source_image)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = Path(tmp) / "boot-test.img"
        shutil.copyfile(source_image, image_path)
        image = FATImage(image_path)
        try:
            assert image.bps in {512, 1024, 2048, 4096}
            assert image.cluster_count > 0
            image.list_dir(None)
            image.write_file("ICC/TST", b"fat image probe")
            assert image.read_file("ICC/TST") == b"fat image probe"
            assert normalize_dos_text(b"a\r\nb\rc\n") == b"a\nb\nc\n"
            image.resolve_parent("ICC/TST")
            tree_source = tmp_path / "tree"
            tree_source.mkdir()
            (tree_source / "old.tmp").write_bytes(b"tree probe")
            put_tree(image, tree_source, "TREE")
            assert image.read_file("TREE/old.tmp") == b"tree probe"
            (tree_source / "old.tmp").unlink()
            (tree_source / "new.tmp").write_bytes(b"replacement probe")
            put_tree(image, tree_source, "TREE")
            assert image.read_file("TREE/new.tmp") == b"replacement probe"
            try:
                image.read_file("TREE/old.tmp")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("put_tree left stale file in replacement directory")
        finally:
            image.close()
    print("trace_scope fat_image_contract")
    print("trace FATImage._find_base")
    print("trace FATImage._read_bpb")
    print("trace FATImage.fat_get")
    print("trace FATImage.fat_set_one")
    print("trace FATImage.alloc_clusters")
    print("trace FATImage.cluster_chain")
    print("trace FATImage.short_name")
    print("trace FATImage.dir_region")
    print("trace FATImage.list_dir")
    print("trace FATImage.find_entry")
    print("trace FATImage.free_slot")
    print("trace FATImage.write_entry")
    print("trace FATImage.mkdir_at")
    print("trace FATImage.resolve_parent")
    print("trace FATImage.write_file")
    print("trace FATImage.read_file")
    print("trace put_tree")
    print("trace normalize_dos_text")
    print("PROBE_OK fat_image_put self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--put", nargs=2, action="append", metavar=("SOURCE", "DEST"), default=[])
    parser.add_argument("--put-tree", nargs=2, action="append", metavar=("SOURCE_DIR", "DEST_DIR"), default=[])
    parser.add_argument("--get", nargs=2, action="append", metavar=("SOURCE", "DEST"), default=[])
    parser.add_argument("--get-text", nargs=2, action="append", metavar=("SOURCE", "DEST"), default=[])
    parser.add_argument("--remove", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.image is None:
        parser.error("image is required unless --self-test is used")

    image = FATImage(args.image)
    try:
        for path in args.remove:
            image.remove(path)
        for source, dest in args.put_tree:
            put_tree(image, Path(source), dest)
        for source, dest in args.put:
            image.write_file(dest, Path(source).read_bytes())
        for source, dest in args.get:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(image.read_file(source))
        for source, dest in args.get_text:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(normalize_dos_text(image.read_file(source)))
    finally:
        image.close()


if __name__ == "__main__":
    main()
