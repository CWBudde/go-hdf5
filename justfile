# go-hdf justfile
# Development automation for pure Go HDF5 library

set shell := ["bash", "-uc"]

# Note: Install dependencies manually or use the GitHub Actions workflow
# treefmt: Download from https://github.com/numtide/treefmt/releases
# Go tools: go install mvdan.cc/gofumpt@latest && go install github.com/daixiang0/gci@latest && go install mvdan.cc/sh/v3/cmd/shfmt@latest
# golangci-lint: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
# prettier: npm install -g prettier

# Default recipe - show available commands
default:
    @just --list

# Format all code using treefmt
fmt:
    treefmt --allow-missing-formatter

# Check if code is formatted correctly
check-formatted:
    treefmt --allow-missing-formatter --fail-on-change

# Run linters
lint:
    golangci-lint run --timeout=5m

# Run linters with auto-fix
lint-fix:
    golangci-lint run --fix --timeout=5m

# Ensure go.mod is tidy
check-tidy:
    go mod tidy
    git diff --exit-code go.mod go.sum

# Run all tests
test:
    go test -v -timeout 120s ./...

# Run tests with race detector
test-race:
    go test -v -race -timeout 120s ./...

# Run tests with coverage
test-coverage:
    go test -v -timeout 120s -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Run benchmarks
benchmark:
    go test -bench=. -benchmem ./...

# Run all checks (formatting, linting, tests, tidiness)
check: check-formatted lint test check-tidy

# Build library (check compilation)
build:
    go build ./...

# Build all examples
examples:
    go build ./examples/...

# Build dump_hdf5 CLI tool
build-dump:
    go build -o bin/dump_hdf5 ./cmd/dump_hdf5

# Clean build artifacts
clean:
    rm -rf bin/
    rm -f coverage.out coverage.html
