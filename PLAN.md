# go-hdf5 — Fork of meko-christian/go-hdf5

## Goal

Maintain a fork of [meko-christian/go-hdf5](https://github.com/meko-christian/go-hdf5) at
[MeKo-Christian/go-hdf5](https://github.com/MeKo-Christian/go-hdf5) as
the HDF5 foundation for our projects (notably
[go-sofa](../go-sofa/)).

## Prior Art & Key Resources

| Resource | Notes |
|----------|-------|
| `../PasHdf/Source/HdfFile.pas` | Our own HDF5 reader (~2 000 LOC Pascal). Read-only, superblock v0-v3, fractal heaps, chunked+compressed datasets. |
| [meko-christian/go-hdf5](https://github.com/meko-christian/go-hdf5) (upstream) | Pure-Go HDF5 v0.13. Read+write, all layouts, gzip/lzf, hyperslab, 86 % coverage. |
| [HDF5 Format Spec v3.0](https://docs.hdfgroup.org/archive/support/HDF5/doc/H5.format.html) | Official binary layout reference. |

## Repository Setup

```text
origin    git@github.com:MeKo-Christian/go-hdf5.git
upstream  git@github.com:meko-christian/go-hdf5.git
```

Module: `github.com/meko-christian/go-hdf5` (keep upstream module path for now;
rename to `github.com/MeKo-Christian/go-hdf5` once we diverge enough).

---

## Phase 0 — Orientation & Housekeeping

**Goal:** Understand the fork's existing API, ensure everything builds
and passes tests.

- [ ] Run `go build ./...` and `go test ./...` — fix anything broken.
- [ ] Read through `file.go`, `group.go`, key exported types (`File`,
      `Group`, `Dataset`, `Attribute`) to understand the public API.
- [ ] Inventory existing `cmd/dump_hdf5` tool.
- [ ] Decide whether to rename the Go module now or keep
      `github.com/meko-christian/go-hdf5` temporarily.
- [ ] Document the upstream sync strategy (how/when to pull from
      scigolib).

---

## Phase 1 — Polish & CI

**Goal:** Production-quality fork maintenance.

### Tasks

- [ ] `go vet` / `golangci-lint` clean.
- [ ] GitHub Actions workflow: build, test, lint.
- [ ] Verify existing test suite passes fully.

---

## Phase 2 — Upstream Contributions & Divergence

**Goal:** Give back fixes, manage fork divergence.

### Tasks

- [ ] Open PRs upstream for any bugs found.
- [ ] If module rename is needed, do it here
      (`github.com/MeKo-Christian/go-hdf5`), update all imports.
- [ ] Document fork-specific changes in `UPSTREAM.md`.

---

## Phase 3 — Optional / Future

- [ ] **Additional filters**: szip, n-bit, scale-offset (if needed).
- [ ] **Performance**: benchmark and optimize hot paths.
- [ ] **HDF4 support** (separate package, if ever needed).

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Upstream API changes | Medium | Pin to known good commit; periodic controlled sync. |
| Module path rename friction | Low | Delay rename until truly needed. |
