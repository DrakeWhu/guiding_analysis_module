#!/usr/bin/env bash
# Configure, build and test the C++ tools with one command.
#
#   tools/build.sh                 # release build + tests
#   tools/build.sh hpc             # compute nodes: optimised, no GUI
#   tools/build.sh dev             # debug build with ASan/UBSan
#   tools/build.sh release --no-tests
#
# HDF5 is the only dependency that is not fetched automatically. Point at it
# with HDF5_ROOT=..., or pass GUIDING_FETCH_HDF5=ON to build it from source.
set -euo pipefail

preset="${1:-release}"
shift || true
run_tests=1
for argument in "$@"; do
  [ "$argument" = "--no-tests" ] && run_tests=0
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

configure=(cmake --preset "$preset")
[ -n "${HDF5_ROOT:-}" ] && configure+=("-DHDF5_ROOT=$HDF5_ROOT")
[ -n "${GUIDING_FETCH_HDF5:-}" ] && configure+=("-DGUIDING_FETCH_HDF5=$GUIDING_FETCH_HDF5")

echo "==> configure ($preset)"
"${configure[@]}"
echo "==> build"
cmake --build --preset "$preset" -j "$(nproc 2>/dev/null || echo 4)"
if [ "$run_tests" = 1 ]; then
  echo "==> test"
  ctest --preset "$preset"
fi

binary_dir="build/$preset"
echo
echo "Built:"
[ -x "$binary_dir/cpp/cli/guiding_cli" ] && echo "  $binary_dir/cpp/cli/guiding_cli   (headless reductions and scores)"
[ -x "$binary_dir/cpp/gui/guiding_gui" ] && echo "  $binary_dir/cpp/gui/guiding_gui   (dashboard)"
echo
echo "Try:  $binary_dir/cpp/cli/guiding_cli campaign --campaign-root CAMPAIGN_DIR"
echo "      $binary_dir/cpp/gui/guiding_gui --campaign-root CAMPAIGN_DIR"
