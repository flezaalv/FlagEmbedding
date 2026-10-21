#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/m3_single_device.py"
LOG_ROOT="${M3_LOG_ROOT:-/logs/logs_${SCRIPT_NAME}}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
ARCHITECTURE="$(uname -m)"
KERNEL_PMU_ARCHITECTURE="$(cat /sys/devices/cpu/caps/pmu_name 2>/dev/null || true)"
KERNEL_PMU_TAG="${KERNEL_PMU_ARCHITECTURE:-unknown_pmu}"
KERNEL_PMU_TAG="$(printf '%s' "$KERNEL_PMU_TAG" | sed -E 's/[^A-Za-z0-9_.-]+/_/g')"
RUN_DIR="$LOG_ROOT/${TIMESTAMP}_${KERNEL_PMU_TAG}"
ARCHITECTURE_LOG="$RUN_DIR/architecture_info.txt"

mkdir -p "$LOG_ROOT" || exit 1
mkdir "$RUN_DIR" || exit 1

CPU_VENDOR="$(awk -F: '/^vendor_id/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
CPU_MODEL_NAME="$(awk -F: '/^model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
CPU_FAMILY="$(awk -F: '/^cpu family/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
CPU_MODEL_ID="$(awk -F: '/^model[ \t]/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
CPU_STEPPING="$(awk -F: '/^stepping/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
CPU_FEATURES="$(awk -F: '/^(flags|Features)/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"

has_feature() {
    case " $CPU_FEATURES " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

{
    printf 'timestamp=%s\n' "$TIMESTAMP"
    printf 'architecture_uname=%s\n' "$ARCHITECTURE"
    printf 'cpu_vendor=%s\n' "${CPU_VENDOR:-unknown}"
    printf 'cpu_model_name=%s\n' "${CPU_MODEL_NAME:-unknown}"
    printf 'cpu_family=%s\n' "${CPU_FAMILY:-unknown}"
    printf 'cpu_model=%s\n' "${CPU_MODEL_ID:-unknown}"
    printf 'cpu_stepping=%s\n' "${CPU_STEPPING:-unknown}"
    printf 'kernel_pmu_architecture=%s\n' "${KERNEL_PMU_ARCHITECTURE:-unknown}"
    printf '\nRelevant ISA support:\n'
    for feature in avx avx2 avx512f avx512bw avx512_vnni avx512_bf16 amx_bf16 amx_int8 f16c; do
        if has_feature "$feature"; then
            printf '%-16s yes\n' "$feature"
        else
            printf '%-16s no\n' "$feature"
        fi
    done
    printf '\nRaw CPU flags/features:\n%s\n' "${CPU_FEATURES:-unknown}"
} > "$ARCHITECTURE_LOG"

printf 'Logs: %s\n' "$RUN_DIR"

for precision in fp32 bf16 fp16; do
    combined_log="$RUN_DIR/${precision}_combined.log"
    onednn_log="$RUN_DIR/${precision}_onednn.log"
    application_log="$RUN_DIR/${precision}_application.log"

    : > "$onednn_log"
    : > "$application_log"

    printf '\n===== %s =====\n' "$precision"
    (
        cd "$PROJECT_ROOT" || exit 1
        M3_PRECISION="$precision" \
        PYTHONUNBUFFERED=1 \
        ONEDNN_VERBOSE=1 \
        python -u "$PYTHON_SCRIPT" 2>&1
    ) | tee "$combined_log" | awk \
        -v onednn="$onednn_log" \
        -v application="$application_log" \
        '{
            if ($0 ~ /^onednn_verbose/) {
                print >> onednn
            } else {
                print >> application
            }
            print
        }'

    status="${PIPESTATUS[0]}"
    if [[ "$status" -ne 0 ]]; then
        printf 'ERROR: %s failed with exit code %s\n' "$precision" "$status" >&2
        exit "$status"
    fi

    printf '\noneDNN ISA (%s):\n' "$precision" >> "$ARCHITECTURE_LOG"
    grep 'onednn_verbose,v1,info,cpu,isa:' "$onednn_log" >> "$ARCHITECTURE_LOG" || \
        printf 'not reported\n' >> "$ARCHITECTURE_LOG"
done

printf '\nLogs and results saved in: %s\n' "$RUN_DIR"