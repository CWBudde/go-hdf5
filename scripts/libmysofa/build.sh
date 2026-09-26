#!/usr/bin/env bash
# Builds the libmysofa load harness (load.c) without cmake.
#
# Usage: scripts/libmysofa/build.sh <libmysofa checkout> <output binary>
#
# Needs a C compiler and zlib. Then run the tests with
#   LIBMYSOFA_LOAD=<output binary> go test -run TestLibmysofaLoad .
set -euo pipefail

if [[ $# -ne 2 ]]; then
	echo "usage: $0 <libmysofa checkout> <output binary>" >&2
	exit 2
fi
src="$1/src"
out="$2"
here="$(cd "$(dirname "$0")" && pwd)"
gen="$(mktemp -d)"
trap 'rm -rf "$gen"' EXIT

# Headers cmake would generate from config.h.in and mysofa_export.h.in.
cat >"$gen/config.h" <<'EOF'
#ifndef _CONFIG_H
#define _CONFIG_H
#define CMAKE_INSTALL_PREFIX "/usr/local"
#define CPACK_PACKAGE_VERSION_MAJOR 0
#define CPACK_PACKAGE_VERSION_MINOR 0
#define CPACK_PACKAGE_VERSION_PATCH 0
#endif
EOF
printf '#define MYSOFA_EXPORT\n' >"$gen/mysofa_export.h"

"${CC:-cc}" -O1 -w -I"$gen" -I"$src/hrtf" -I"$src" -o "$out" \
	"$here/load.c" "$src"/hrtf/*.c "$src"/hdf/*.c "$src/resampler/speex_resampler.c" -lz -lm
