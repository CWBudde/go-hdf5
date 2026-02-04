# go-hdf5 — Fork of meko-christian/go-hdf5

## Goal

Maintain a fork of [meko-christian/go-hdf5](https://github.com/meko-christian/go-hdf5) at
[MeKo-Christian/go-hdf5](https://github.com/MeKo-Christian/go-hdf5) as
the HDF5 foundation for our projects (notably
[go-sofa](../go-sofa/)).

## Prior Art & Key Resources

| Resource                                                                                   | Notes                                                                                                             |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `../PasHdf/Source/HdfFile.pas`                                                             | Our own HDF5 reader (~2 000 LOC Pascal). Read-only, superblock v0-v3, fractal heaps, chunked+compressed datasets. |
| [meko-christian/go-hdf5](https://github.com/meko-christian/go-hdf5) (upstream)             | Pure-Go HDF5 v0.13. Read+write, all layouts, gzip/lzf, hyperslab, 86 % coverage.                                  |
| [HDF5 Format Spec v3.0](https://docs.hdfgroup.org/archive/support/HDF5/doc/H5.format.html) | Official binary layout reference.                                                                                 |

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

- [x] Run `go build ./...` and `go test ./...` — fix anything broken.
- [x] Decide whether to rename the Go module now or keep
      `github.com/meko-christian/go-hdf5` temporarily.
- [ ] Read through `file.go`, `group.go`, key exported types (`File`,
      `Group`, `Dataset`, `Attribute`) to understand the public API.
- [ ] Inventory existing `cmd/dump_hdf5` tool.

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

## Phase 3 — Root Group Attributes Enhancement 🚨 CRITICAL

**Status**: 🔴 Blocking go-sofa write support

**Goal**: Enable writing attributes to root "/" group during file creation to support SOFA file format requirements.

### Problem Statement

**Current Limitation**: Root group object header is created with a fixed size during file initialization (`CreateForWrite`) and cannot accommodate attributes added later. Attempting to add attributes post-creation causes file corruption.

**Impact**: Blocks SOFA file writing, which requires ~20 global attributes on the root group per AES69 specification. Also affects any application needing to write root-level metadata (netCDF-4, HDF5-based scientific formats).

### Technical Analysis

#### Current Behavior

1. **File Creation** ([dataset_write.go:671-787](dataset_write.go))

   ```
   CreateForWrite() →
   └─ createRootGroupStructureV2()
      ├─ Creates root group object header at offset 48
      │  └─ Size: ~100 bytes (Symbol Table message ONLY)
      └─ Writes adjacent structures (local heap, symbol table, B-tree)
   ```

2. **Attribute Writing Attempt** ([attribute_write.go:205-241](attribute_write.go))

   ```
   RootGroup().WriteAttribute() →
   └─ writeAttribute()
      ├─ Reads object header from offset 48 (~100 bytes)
      ├─ Adds attribute message → header grows to ~500 bytes
      ├─ Writes modified header back to SAME offset 48
      └─ ❌ OVERWRITES local heap/symbol table → CORRUPTION
   ```

3. **File Corruption Symptoms**
   - Created files cannot be reopened: `EOF` errors
   - h5dump fails: `h5dump error: internal error`
   - Root group load fails: `root group load failed: object header read failed`

#### Root Cause

HDF5 object headers have a **fixed size at creation**. The current implementation:

- Creates root group with minimal header (Symbol Table message only)
- Provides no mechanism to pre-allocate space for attributes
- `writeAttribute()` modifies header in-place, causing overflow

### Solution: WithRootAttributes API

Add functional options to `CreateForWrite()` allowing attributes to be specified during file creation, enabling proper header size calculation.

#### API Design

```go
// In dataset_write.go
type CreateOption func(*createOptions)

type createOptions struct {
    rootAttributes map[string]interface{}
}

func WithRootAttribute(name string, value interface{}) CreateOption {
    return func(opts *createOptions) {
        if opts.rootAttributes == nil {
            opts.rootAttributes = make(map[string]interface{})
        }
        opts.rootAttributes[name] = value
    }
}

func CreateForWrite(filename string, mode CreateMode, opts ...CreateOption) (*FileWriter, error) {
    // Apply options
    config := &createOptions{}
    for _, opt := range opts {
        opt(config)
    }

    // Pass attributes to root group creation
    // ...
}
```

#### Implementation Tasks

### ✅ Phase 3.1: API Design (30 minutes)

- [ ] **Design CreateOption pattern**
  - [ ] Define `CreateOption` func type
  - [ ] Define `createOptions` struct with `rootAttributes` field
  - [ ] Create `WithRootAttribute(name, value)` option constructor
  - [ ] Update `CreateForWrite()` signature to accept `...CreateOption`

- [ ] **Backward Compatibility**
  - [ ] Ensure existing calls to `CreateForWrite(path, mode)` work unchanged
  - [ ] Default behavior: no attributes, same as current

- [ ] **Documentation**
  - [ ] Add godoc for `CreateOption`, `WithRootAttribute`
  - [ ] Add example to package documentation
  - [ ] Update README with root attribute example

### ✅ Phase 3.2: Compact Attribute Storage (2-3 hours)

**Goal**: Support ≤8 root attributes stored directly in object header

- [ ] **Header Size Calculation** ([dataset_write.go:2668-2710](dataset_write.go))
  - [ ] Add `calculateAttributeMessagesSize(attrs map[string]interface{}) int`
  - [ ] Calculate size for each attribute (name, datatype, dataspace, value)
  - [ ] Account for message headers and alignment
  - [ ] Return total size needed

- [ ] **Modify createRootGroupStructureV2**
  - [ ] Accept `attrs map[string]interface{}` parameter
  - [ ] Calculate total header size: SymbolTable + AttributeMessages
  - [ ] Pre-allocate correct header size before writing
  - [ ] Write Symbol Table message first
  - [ ] Write attribute messages inline (compact storage)
  - [ ] Write header continuation if needed (>~256 bytes)

- [ ] **Update writeRootGroupHeader** ([dataset_write.go:2893-2928](dataset_write.go))
  - [ ] Accept `attrs` parameter
  - [ ] Build message list: Symbol Table + Attribute messages
  - [ ] Use existing `encodeAttributeMessage()` for each attribute
  - [ ] Write complete header with all messages at once

- [ ] **Testing**
  - [ ] Test with 0 attributes (baseline - existing behavior)
  - [ ] Test with 1 attribute (simple case)
  - [ ] Test with 5 attributes (typical case)
  - [ ] Test with 8 attributes (compact storage limit)
  - [ ] Verify with h5dump for each case
  - [ ] Round-trip test: create → close → reopen → verify attributes

### ✅ Phase 3.3: Dense Attribute Storage (2-3 hours)

**Goal**: Support >8 root attributes using Fractal Heap + B-tree

- [ ] **Dense Storage Detection**
  - [ ] Define threshold constant: `MaxCompactAttributes = 8`
  - [ ] Check `len(attrs) > MaxCompactAttributes` in header creation

- [ ] **Fractal Heap Creation** ([attribute_write.go](attribute_write.go))
  - [ ] Extract/create `createAttributeFractalHeap()` function
  - [ ] Write Fractal Heap header (direct block, heap metadata)
  - [ ] Return heap address and size

- [ ] **Attribute Name Index B-tree**
  - [ ] Extract/create `createAttributeNameIndex()` function
  - [ ] Create B-tree Type 8 (attribute name index)
  - [ ] Populate with attribute name → heap ID mappings
  - [ ] Return B-tree root address

- [ ] **Attribute Info Message**
  - [ ] Create `encodeAttributeInfoMessage()` function
  - [ ] Include Fractal Heap address
  - [ ] Include B-tree name index address
  - [ ] Set MaxCreationIndex (number of attributes)
  - [ ] Return encoded message

- [ ] **Update writeRootGroupHeader for Dense Storage**
  - [ ] If `len(attrs) > 8`:
    - [ ] Create Fractal Heap for attributes
    - [ ] Create B-tree name index
    - [ ] Write Attribute Info message (NOT individual attribute messages)
    - [ ] Store heap/B-tree addresses in message
  - [ ] Else: use compact storage (Phase 3.2)

- [ ] **Testing**
  - [ ] Test with 9 attributes (triggers dense storage)
  - [ ] Test with 20 attributes (SOFA use case)
  - [ ] Test with 50 attributes (stress test)
  - [ ] Verify with h5dump
  - [ ] Round-trip tests for all cases
  - [ ] Verify reading attributes works (existing code)

### ✅ Phase 3.4: Integration & Testing (1 hour)

- [ ] **Update CreateForWrite**
  - [ ] Wire `createOptions.rootAttributes` through to `createRootGroupStructureV2`
  - [ ] Handle empty attributes map (backward compatibility)

- [ ] **Comprehensive Test Suite**
  - [ ] `TestCreateWithRootAttributes` - parametric test:
    - [ ] 0, 1, 5, 8 attributes (compact)
    - [ ] 9, 20, 50 attributes (dense)
  - [ ] `TestRootAttributeRoundTrip` - for each count:
    - [ ] Create file with N attributes
    - [ ] Close file
    - [ ] Reopen file
    - [ ] Read root attributes
    - [ ] Verify all values match
  - [ ] `TestRootAttributeTypes` - various datatypes:
    - [ ] String (fixed + variable-length)
    - [ ] Integer (int32, int64)
    - [ ] Float (float32, float64)
    - [ ] Array types
  - [ ] `TestH5DumpCompatibility`:
    - [ ] Create file with root attributes
    - [ ] Run h5dump via exec.Command
    - [ ] Parse output, verify attributes present

- [ ] **Error Handling**
  - [ ] Invalid attribute names (empty, invalid chars)
  - [ ] Unsupported attribute datatypes
  - [ ] Attribute value encoding errors
  - [ ] File write errors during header creation

- [ ] **Performance Testing**
  - [ ] Benchmark attribute writing (1, 10, 50, 100 attributes)
  - [ ] Memory profiling for large attribute counts
  - [ ] Compare compact vs dense storage overhead

### ✅ Phase 3.5: Documentation & Examples (30 minutes)

- [ ] **Update README.md**
  - [ ] Add "Writing Root Attributes" section
  - [ ] Example: Create file with root metadata
  - [ ] Example: netCDF-4 / SOFA file creation pattern

- [ ] **Create Example** ([examples/09-root-attributes/](examples/09-root-attributes/))
  - [ ] `main.go` - demonstrate WithRootAttribute usage
  - [ ] Create file with 5 attributes
  - [ ] Reopen and verify
  - [ ] Show compact and dense storage examples
  - [ ] README.md with explanation

- [ ] **Update Architecture Docs** ([docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md))
  - [ ] Document root attribute creation flow
  - [ ] Explain compact vs dense storage decision
  - [ ] Diagram showing file layout with root attributes

- [ ] **GoDoc Enhancements**
  - [ ] Package-level example using WithRootAttribute
  - [ ] Document compact vs dense storage threshold
  - [ ] Link to HDF5 specification sections

### Success Criteria

1. ✅ **API Completeness**: `WithRootAttribute()` option works for any number of attributes
2. ✅ **Storage Modes**: Both compact (≤8) and dense (>8) storage implemented
3. ✅ **Backward Compatibility**: Existing code works unchanged (no attributes = current behavior)
4. ✅ **File Validity**: Created files pass h5dump validation
5. ✅ **Round-Trip**: Files can be reopened and attributes read correctly
6. ✅ **SOFA Compatibility**: 20+ attributes work (blocks go-sofa write support)
7. ✅ **Test Coverage**: ≥80% coverage for new code
8. ✅ **Documentation**: Clear examples and API docs

### Files to Modify

| File                                                                       | Changes Required                                                           | Complexity |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------- |
| [dataset_write.go](dataset_write.go)                                       | Add CreateOption, modify CreateForWrite, update createRootGroupStructureV2 | High       |
| [attribute_write.go](attribute_write.go)                                   | Extract/create Fractal Heap and B-tree helpers for dense storage           | Medium     |
| [internal/core/objectheader_write.go](internal/core/objectheader_write.go) | May need attribute message encoding helpers                                | Low        |
| [group_write_test.go](group_write_test.go)                                 | New test cases for root attributes                                         | Medium     |
| [README.md](README.md)                                                     | Add root attribute examples                                                | Low        |
| [examples/09-root-attributes/](examples/)                                  | New example demonstrating feature                                          | Low        |

### Dependencies

- **go-sofa**: Blocked waiting for this enhancement
- **Upstream**: Consider PR to meko-christian/go-hdf5 after validation
- **HDF5 Spec**: [Object Header Format](https://docs.hdfgroup.org/hdf5/develop/_s_p_e_c.html#OHDRLayout), [Attribute Message](https://docs.hdfgroup.org/hdf5/develop/_s_p_e_c.html#AttributeMessage)

### Testing Strategy

1. **Unit Tests**: Individual functions (header size calculation, message encoding)
2. **Integration Tests**: Full file creation → close → reopen → verify cycle
3. **Validation Tests**: h5dump output parsing and verification
4. **Round-Trip Tests**: Write attributes → read back → compare
5. **Performance Tests**: Benchmark creation time and memory usage

### Risk Assessment

| Risk                                    | Probability | Impact | Mitigation                                            |
| --------------------------------------- | ----------- | ------ | ----------------------------------------------------- |
| Header size miscalculation              | Medium      | High   | Extensive testing with various attribute counts/types |
| Dense storage implementation complexity | Medium      | High   | Start with compact, add dense incrementally           |
| Fractal Heap creation bugs              | Medium      | High   | Reuse existing Fractal Heap code, test thoroughly     |
| Backward compatibility break            | Low         | High   | Functional options pattern preserves existing API     |
| Performance regression                  | Low         | Medium | Benchmark before/after, optimize if needed            |

### Estimated Timeline

- **Phase 3.1 (API Design)**: 30 minutes
- **Phase 3.2 (Compact Storage)**: 2-3 hours
- **Phase 3.3 (Dense Storage)**: 2-3 hours
- **Phase 3.4 (Testing)**: 1 hour
- **Phase 3.5 (Documentation)**: 30 minutes

**Total**: 6-8 hours for complete implementation and validation

### Next Steps After Completion

1. **Update go-sofa**: Modify `Save()` to use `WithRootAttribute()` API
2. **Run go-sofa Tests**: Verify SOFA file creation works end-to-end
3. **Upstream PR**: Consider contributing this enhancement to meko-christian/go-hdf5
4. **Documentation**: Update go-hdf5 CHANGELOG and release notes

---

## Phase 4 — Optional / Future

- [ ] **Additional filters**: szip, n-bit, scale-offset (if needed).
- [ ] **Performance**: benchmark and optimize hot paths.
- [ ] **HDF4 support** (separate package, if ever needed).

---

## Risk Register

| Risk                                     | Impact | Mitigation                                                                           |
| ---------------------------------------- | ------ | ------------------------------------------------------------------------------------ |
| Upstream API changes                     | Medium | Pin to known good commit; periodic controlled sync.                                  |
| Module path rename friction              | Low    | Delay rename until truly needed.                                                     |
| Root attribute implementation complexity | High   | Incremental implementation: compact first, then dense. Test extensively with h5dump. |
