#!/usr/bin/env bash
# Prints (or runs) the QEMU command that boots one published shape of a KunoWorld CVM release as a TD.
#
#   image/cvm/launch-td.sh <release dir> <shape id> --gpu <PCI address>... --weights <name>=<image prefix>...
#       [--env worker.env] [--hotkey-seed hotkey.seed] [--qgs-port 4050] [--qemu qemu-system-x86_64] [--run]
#
#   <release dir>    build.sh output: ovmf.fd, bzImage, initramfs.cpio.gz, rootfs.img.verity, metadata.json
#   --weights        weights-verity.sh output prefix, built with KUNO_WEIGHTS_LAYOUT=appended
#                    (<prefix>.img, <prefix>.roothash, <prefix>.size); <name> becomes /models/<name>
#   --env            KEY=VALUE lines for the worker (kuno-app keeps only its allowlist)
#
# RTMR0 measures the ACPI tables QEMU builds from the devices it exposes. The pinned dstack-mr models
# dstack-vmm's command line (Dstack-TEE/dstack dstack/vmm/src/app/qemu.rs), so this emits the same
# devices in the same order: the verity root disk, a data disk, one virtio disk per weights image
# (num_verity_volumes), one NIC, a vsock device, then each GPU behind its own pcie-root-port on iommufd.
# Anything else (a second NIC, a TPM, a shared folder, hugepages, memory hotplug) changes RTMR0 and the
# TD will not match the manifest. NOT RUN ON A TDX HOST: confirm the first boot with publish.py compare-quote.
set -euo pipefail

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ $# -ge 2 ] || usage
release="$(cd "$1" && pwd)"
shape="$2"
shift 2
here="$(cd "$(dirname "$0")" && pwd)"
qemu="qemu-system-x86_64"
qgs_port=4050
run=0
env_file=""
seed_file=""
declare -a gpus=() weights=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpus+=("${2:?--gpu needs a PCI address}"); shift 2 ;;
    --weights) weights+=("${2:?--weights needs name=prefix}"); shift 2 ;;
    --env) env_file="${2:?}"; shift 2 ;;
    --hotkey-seed) seed_file="${2:?}"; shift 2 ;;
    --qgs-port) qgs_port="${2:?}"; shift 2 ;;
    --qemu) qemu="${2:?}"; shift 2 ;;
    --run) run=1; shift ;;
    *) usage ;;
  esac
done

# shape fields, one per line: cpus memory_mib num_gpus num_nvswitches num_nics num_verity_volumes hugepages hotplug_off pci_hole64_size
mapfile -t fields < <(python3 - "$here" "$shape" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import measure
s = measure.load_shape(f"{sys.argv[1]}/shapes.json:{sys.argv[2]}")
for value in (s.cpus, s.memory // 2**20, s.num_gpus, s.num_nvswitches, s.num_nics, s.num_verity_volumes,
              int(s.hugepages), int(s.hotplug_off), s.pci_hole64_size or 0):
    print(value)
PY
)
cpus="${fields[0]}"; memory_mib="${fields[1]}"; num_gpus="${fields[2]}"; num_nvswitches="${fields[3]}"
num_nics="${fields[4]}"; num_volumes="${fields[5]}"; hugepages="${fields[6]}"; hotplug_off="${fields[7]}"; hole64="${fields[8]}"

[ "${#gpus[@]}" -eq "$num_gpus" ] || { echo "$shape needs $num_gpus GPU(s); got ${#gpus[@]}" >&2; exit 1; }
[ "${#weights[@]}" -eq "$num_volumes" ] || { echo "$shape attaches $num_volumes weights disk(s); got ${#weights[@]}" >&2; exit 1; }
[ "$num_nics" -eq 1 ] || { echo "launch-td.sh emits exactly one NIC; $shape declares $num_nics" >&2; exit 1; }
[ "$num_nvswitches" -eq 0 ] || { echo "NVSwitch passthrough is not supported by this launcher" >&2; exit 1; }
[ "$hugepages" -eq 0 ] || { echo "hugepages change the QEMU NUMA layout; not supported by this launcher" >&2; exit 1; }
for file in ovmf.fd bzImage initramfs.cpio.gz rootfs.img.verity metadata.json; do
  [ -f "$release/$file" ] || { echo "$release has no $file" >&2; exit 1; }
done
cmdline="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["cmdline"])' "$release/metadata.json")"

state="$release/launch-$shape"
mkdir -p "$state"
[ -f "$state/data.img" ] || truncate -s 64M "$state/data.img"  # present because dstack-vmm always attaches one; kuno-app never mounts it
: > "$state/weights.txt"

args=(-accel kvm -cpu host -nographic -nodefaults
  -chardev "stdio,id=com0,logfile=$state/serial.log,logappend=on" -serial chardev:com0
  -bios "$release/ovmf.fd" -kernel "$release/bzImage" -initrd "$release/initramfs.cpio.gz")
[ "$hotplug_off" -eq 1 ] && args+=(-global ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off)
[ "$hole64" -gt 0 ] && args+=(-global "q35-pcihost.pci-hole64-size=$(printf '0x%x' "$hole64")")
args+=(-drive "file=$release/rootfs.img.verity,if=none,id=hd0,format=raw,readonly=on" -device "virtio-blk-pci,drive=hd0")
args+=(-drive "file=$state/data.img,if=none,id=hd1,format=raw" -device "virtio-blk-pci,drive=hd1")
index=0
for spec in "${weights[@]}"; do
  name="${spec%%=*}"; prefix="${spec#*=}"
  [[ "$name" =~ ^[a-z0-9][a-z0-9.-]{0,12}$ ]] || { echo "weights name $name must be 1-13 of [a-z0-9.-]" >&2; exit 1; }
  for suffix in img roothash size; do
    [ -f "$prefix.$suffix" ] || { echo "$prefix.$suffix is missing: build it with KUNO_WEIGHTS_LAYOUT=appended weights-verity.sh" >&2; exit 1; }
  done
  printf '%s %s %s\n' "$name" "$(cat "$prefix.roothash")" "$(cat "$prefix.size")" >> "$state/weights.txt"
  args+=(-drive "file=$prefix.img,if=none,id=vol$index,format=raw,readonly=on" -device "virtio-blk-pci,drive=vol$index,serial=kuno-w-$name")
  index=$((index + 1))
done
args+=(-netdev "user,id=net0" -device "virtio-net-pci,netdev=net0")
args+=(-device "vhost-vsock-pci,guest-cid=3")
args+=(-machine "q35,kernel-irqchip=split,confidential-guest-support=tdx,hpet=off")
args+=(-object "{\"qom-type\":\"tdx-guest\",\"id\":\"tdx\",\"quote-generation-socket\":{\"type\":\"vsock\",\"cid\":\"2\",\"port\":\"$qgs_port\"}}")
if [ "$num_gpus" -gt 0 ]; then
  args+=(-object "iommufd,id=iommufd0")
  slot=1
  for gpu in "${gpus[@]}"; do
    args+=(-device "pcie-root-port,id=pci.$slot,bus=pcie.0,chassis=$slot" -device "vfio-pci,host=$gpu,bus=pci.$slot,iommufd=iommufd0")
    slot=$((slot + 1))
  done
fi
args+=(-smp "$cpus" -m "${memory_mib}M")
args+=(-fw_cfg "name=opt/kuno/weights,file=$state/weights.txt")
[ -n "$env_file" ] && args+=(-fw_cfg "name=opt/kuno/env,file=$env_file")
[ -n "$seed_file" ] && args+=(-fw_cfg "name=opt/kuno/hotkey.seed,file=$seed_file")
args+=(-append "$cmdline")

if [ "$run" = 1 ]; then
  exec "$qemu" "${args[@]}"
fi
printf '%q ' "$qemu" "${args[@]}"
printf '\n'
