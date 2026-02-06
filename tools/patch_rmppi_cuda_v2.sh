#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/src/R-MPPI_F1tenth1"
FILE="${REPO}/python/bindings/rmppi_cuda.cu"
BUILD="${REPO}/build_minisim"

[[ -f "${FILE}" ]] || { echo "ERROR: Missing ${FILE}"; exit 1; }

ts="$(date +%Y%m%d_%H%M%S)"
cp -v "${FILE}" "${FILE}.bak_${ts}"

echo "Backed up -> ${FILE}.bak_${ts}"
echo "Patching ${FILE} ..."

python3 - <<'PY'
import re
from pathlib import Path

path = Path.home()/ "src"/"R-MPPI_F1tenth1"/"python"/"bindings"/"rmppi_cuda.cu"
txt = path.read_text()

changes = []

def sub_exact(old, new, desc):
    global txt
    if old in txt:
        txt = txt.replace(old, new)
        changes.append(desc)

def sub_regex(pat, rep, desc, flags=0):
    global txt
    new, n = re.subn(pat, rep, txt, flags=flags)
    if n:
        txt = new
        changes.append(f"{desc} (x{n})")

# --- Fix 1: remove AX/AY lines (your exact errors) ---
# Delete the two assignments entirely
txt_before = txt
txt = re.sub(r'^\s*sp\.std_dev\[\(int\)DoubleIntegratorParams::ControlIndex::AX\]\s*=\s*1\.0f;\s*\n', '', txt, flags=re.M)
txt = re.sub(r'^\s*sp\.std_dev\[\(int\)DoubleIntegratorParams::ControlIndex::AY\]\s*=\s*1\.0f;\s*\n', '', txt, flags=re.M)
if txt != txt_before:
    changes.append("Removed DoubleIntegratorParams::ControlIndex::AX/AY std_dev assignments")

# If there is NOT already a size-based initialization, insert one after first mention of sp.std_dev
if "initialize std_dev for all control dimensions" not in txt:
    m = re.search(r'(sp\.std_dev\s*\[.*?\]\s*=\s*.*?;\s*)', txt)
    if m:
        insert = (
            "\n    // Robust: initialize std_dev for all control dimensions\n"
            "    for (int i = 0; i < (int)sp.std_dev.size(); i++) sp.std_dev[i] = 1.0f;\n"
        )
        txt = txt[:m.end(1)] + insert + txt[m.end(1):]
        changes.append("Inserted size-based std_dev init loop")

# --- Fix 2: DynamicsParams namespace (your exact error line) ---
sub_regex(r'\bstruct\s+KinematicBicycleParams\s*:\s*public\s+MPPI_internal::DynamicsParams\b',
          'struct KinematicBicycleParams : public DynamicsParams',
          "KinematicBicycleParams now inherits DynamicsParams (not MPPI_internal::DynamicsParams)")

# Also replace any remaining MPPI_internal::DynamicsParams tokens
sub_regex(r'\bMPPI_internal::DynamicsParams\b', 'DynamicsParams',
          "Replace MPPI_internal::DynamicsParams -> DynamicsParams")

# --- Fix 3: StateIndex::VEL -> StateIndex::V ---
sub_regex(r'KinematicBicycleParams::StateIndex::VEL\b', 'KinematicBicycleParams::StateIndex::V',
          "Replace StateIndex::VEL -> StateIndex::V")

# --- Fix 4: wheelbase/tau_speed param names used in your errors ---
# Replace both read-side and write-side usage
sub_regex(r'\bp\.wheelbase\b', 'p.wheelbase_m', "Replace p.wheelbase -> p.wheelbase_m")
sub_regex(r'\bp\.tau_speed\b', 'p.tau_speed_s', "Replace p.tau_speed -> p.tau_speed_s")

# --- Fix 5: Cost base class ctor PARENT_CLASS(stream) -> PARENT_CLASS() ---
sub_regex(r'PARENT_CLASS\s*\(\s*stream\s*\)', 'PARENT_CLASS()',
          "Replace PARENT_CLASS(stream) -> PARENT_CLASS()")

# --- Fix 6: KinematicBicycleDynamics is abstract because stateFromMap not overridden ---
# We inject a correct override if missing.
has_kbd = re.search(r'\b(class|struct)\s+KinematicBicycleDynamics\b', txt) is not None
has_good_override = re.search(
    r'state_array\s+stateFromMap\s*\(\s*const\s+std::map\s*<\s*std::string\s*,\s*float\s*>\s*&\s*\w+\s*\)\s*override',
    txt
) is not None

if has_kbd and not has_good_override:
    # Insert just before the closing brace of the KinematicBicycleDynamics class/struct.
    # We locate the class block by finding "KinematicBicycleDynamics" then the next "};"
    start = re.search(r'\b(class|struct)\s+KinematicBicycleDynamics\b', txt).start()
    end = txt.find("};", start)
    if end != -1:
        insert = (
            "\n  // Required override: convert named map -> state vector\n"
            "  state_array stateFromMap(const std::map<std::string, float>& map) override {\n"
            "    state_array s;\n"
            "    s.setZero();\n"
            "    auto get = [&](const std::string& k, float def) {\n"
            "      auto it = map.find(k);\n"
            "      return (it == map.end()) ? def : it->second;\n"
            "    };\n"
            "    // Common key names in gym / logs\n"
            "    s[(int)KinematicBicycleParams::StateIndex::X]   = get(\"x\", 0.0f);\n"
            "    s[(int)KinematicBicycleParams::StateIndex::Y]   = get(\"y\", 0.0f);\n"
            "    s[(int)KinematicBicycleParams::StateIndex::YAW] = get(\"yaw\", get(\"theta\", 0.0f));\n"
            "    s[(int)KinematicBicycleParams::StateIndex::V]   = get(\"v\", get(\"vx\", 0.0f));\n"
            "    return s;\n"
            "  }\n"
        )
        txt = txt[:end] + insert + txt[end:]
        changes.append("Injected KinematicBicycleDynamics::stateFromMap(map) override")

# Final write
path.write_text(txt)

print("Patch summary:")
for c in changes:
    print(" -", c)

if not changes:
    print(" - (No changes were applied — patterns not found)")
PY

echo
echo "Rebuilding rmppi_cuda ..."
cd "${BUILD}"
rm -f python_build/CMakeFiles/rmppi_cuda.dir/bindings/rmppi_cuda.cu.o || true
make -j4 rmppi_cuda

echo
echo "Built .so files (if any):"
find "${BUILD}/python_build" -maxdepth 6 -name "*.so" -print || true
