# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
just fmt                # Format all code (treefmt: gofumpt, gci, shfmt, prettier)
just check-formatted    # Check formatting (CI)
just lint               # golangci-lint run --timeout=5m
just lint-fix           # Lint with auto-fix
just test               # go test -v -timeout 120s ./...
just test-race          # Tests with race detector
just test-coverage      # Tests with HTML coverage report
just check              # All checks (format + lint + test + tidy)
just build              # go build ./...

# Run a single test
go test -v -run TestName ./path/to/package/

# Run a single benchmark
go test -bench BenchmarkName -benchmem ./path/to/package/
```

Formatting uses `treefmt` (`treefmt.toml`) with gofumpt, gci, shfmt, and prettier. Linting uses golangci-lint with 34+ linters (`.golangci.yml`). Zero lint issues required.

## Module

`github.com/meko-christian/go-hdf5` — pure Go HDF5 library, no CGo. Only test dependency is `testify`.

## Architecture

4-layer design with strict reader/writer file separation (e.g., `file.go` / `file_write.go`):

| Layer      | Location               | Responsibility                                                                                        |
| ---------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Public API | root package           | `File`, `Dataset`, `Group`, functional options                                                        |
| Core       | `internal/core/`       | Superblock, object headers (v1+v2), datatypes, dataspaces, filter pipelines                           |
| Structures | `internal/structures/` | B-trees (v1 read, v2 read+write), fractal heaps, symbol tables, local heaps                           |
| Writer     | `internal/writer/`     | FileWriter, allocator, chunk coordinator, compression filters (gzip, lzf, bzip2, shuffle, fletcher32) |

Additional internal packages:

- `internal/rebalancing/` — B-tree rebalancing strategies (lazy, incremental, smart auto-tuning)
- `internal/utils/` — Buffer pooling (`sync.Pool`), endian-aware I/O, context-rich error wrapping, overflow protection
- `internal/testing/` — Mock readers

## Key Patterns

- **Functional options** for write configuration: `hdf5.CreateForWrite("f.h5", hdf5.CreateTruncate, hdf5.WithLazyRebalancing(...))`
- **Upsert semantics** for attributes: `WriteAttribute` creates or modifies automatically
- **Signature-based dispatch**: 4-byte magic signatures identify structures (SNOD, OHDR, TREE, BTHD, FHDB)
- **Version-specific handling**: Superblock v0/v2/v3, object headers v1/v2, B-trees v1/v2

## Testing

86%+ coverage target (>70% enforced by pre-release check). Test fixtures in `testdata/` with reference files, official HDF5 test suite, and C library corpus. CI runs on Linux, macOS, Windows with Go 1.25.
