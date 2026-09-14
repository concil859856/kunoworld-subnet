#!/usr/bin/env bash
# Prints (or runs) the QEMU command that boots one published shape of a KunoWorld CVM release as a TD.
#
#   image/cvm/launch-td.sh <release dir> <shape id> --gpu <PCI address>... [--nvswitch <PCI address>...]
#       --weights <name>=<image prefix>... [--image <prefix>] [--instance <n>] [--numa-node <n>|auto]
#       [--env worker.env] [--hotkey-seed hotkey.seed] [--qgs-port 4050] [--qemu qemu-system-x86_64] [--run]
#
#   <release dir>    build.sh output: ovmf.fd, bzImage, initramfs.cpio.gz, rootfs.img.verity, metadata.json, worker.*
#   --nvswitch       an NVSwitch of a Protected PCIe shape (exactly its num_nvswitches), bound to vfio-pci
#   --weights        weights-verity.sh output prefix, built with KUNO_WEIGHTS_LAYOUT=appended
#                    (<prefix>.img, <prefix>.roothash, <prefix>.size); <name> becomes /models/<name>
#   --image          the worker image disk, pack-image.sh output (<prefix>.img.verity, .roothash, .size, .digest);
#                    default <release dir>/worker. A Turbo candidate boots the owner's release with its own.
#   --env            KEY=VALUE lines for the worker (kuno-app keeps only its allowlist)
#   --instance       the n-th TD on this host (0-99): guest CID 3+n, state in launch-<shape>.<n>
#                    (without it: CID 3, launch-<shape>); plan-host.py gives one per GPU
#   --numa-node      run QEMU under numactl on host NUMA node n, or auto for the first GPU's node
#                    (read from ${KUNO_SYSFS_ROOT:-/sys}/bus/pci/devices/<gpu>/numa_node)
#
# RTMR0 measures the ACPI tables QEMU builds from the devices it exposes. The pinned dstack-mr models
# dstack-vmm's command line (Dstack-TEE/dstack dstack/vmm/src/app/qemu.rs), so this emits the same
# devices in the same order: the verity root disk, a data disk, then the read-only verity volumes after the data
# disk and before networking (configure_volumes): the worker image disk first, then one virtio disk per weights
# image, together num_verity_volumes; one NIC, a vsock device, then each GPU behind its own pcie-root-port on
# iommufd, then each NVSwitch the same way with the port numbers continuing (dstack-vmm configure_gpus: gpus, then
# bridges; dstack-mr counts num_gpus + num_nvswitches root ports on pcie.0 without hugepages).
# The ACPI model counts volumes and does not see which file, serial or root hash each carries, so every worker
# image boots under the same RTMR0. Anything else (a second NIC, a TPM, a shared folder, hugepages, memory
# hotplug) changes RTMR0 and the TD will not match the manifest. NOT RUN ON A TDX HOST: confirm the first boot
# with publish.py compare-quote.
#
# Several TDs on one host (CVM.md, "Several TDs on one server") differ only in what RTMR0 does not cover:
# - The vsock guest CID. QEMU 9.1 puts it in the device's virtio config space and the VHOST_VSOCK_SET_GUEST_CID
#   ioctl (hw/virtio/vhost-vsock.c) and nowhere in ACPI; the pinned dstack-mr's ACPI model (crates/qemu-acpi
#   MachineConfig) has no CID input; dstack-vmm gives each VM its own CID from a pool (vmm/src/app.rs) under one
#   measurement. Checked in the source only, unverified on a TDX host.
# - --numa-node. numactl binds QEMU's threads and memory on the host and adds no -numa option, so the guest keeps
#   one flat node and no SRAT. Unverified: that the host kernel honours --membind for TD private memory
#   (check numastat -p <qemu pid>).
set -euo pipefail

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
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
instance=""
numa_node=""
image_prefix=""
declare -a gpus=() nvswitches=() weights=() wrapper=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpus+=("${2:?--gpu needs a PCI address}"); shift 2 ;;
    --nvswitch) nvswitches+=("${2:?--nvswitch needs a PCI address}"); shift 2 ;;
    --weights) weights+=("${2:?--weights needs name=prefix}"); shift 2 ;;
    --image) image_prefix="${2:?--image needs a pack-image.sh output prefix}"; shift 2 ;;
    --instance) instance="${2:?--instance needs a number}"; shift 2 ;;
    --numa-node) numa_node="${2:?--numa-node needs a node number or auto}"; shift 2 ;;
    --env) env_file="${2:?}"; shift 2 ;;
    --hotkey-seed) seed_file="${2:?}"; shift 2 ;;
    --qgs-port) qgs_port="${2:?}"; shift 2 ;;
    --qemu) qemu="${2:?}"; shift 2 ;;
    --run) run=1; shift ;;
    *) usage ;;
  esac
done
image_prefix="${image_prefix:-$release/worker}"

cid=3  # 0-2 are reserved; the host, where the QGS listens, is 2
state="$release/launch-$shape"
if [ -n "$instance" ]; then
  [[ "$instance" =~ ^(0|[1-9][0-9]?)$ ]] || { echo "--instance must be an integer from 0 to 99; got $instance" >&2; exit 1; }
  cid=$((3 + instance))
  state="$release/launch-$shape.$instance"
fi
if [ -n "$numa_node" ]; then
  command -v numactl >/dev/null || { echo "--numa-node needs numactl on the host: install it (apt install numactl) or omit --numa-node" >&2; exit 1; }
  sysfs="${KUNO_SYSFS_ROOT:-/sys}"
  if [ "$numa_node" = auto ]; then
    [ "${#gpus[@]}" -gt 0 ] || { echo "--numa-node auto reads the first GPU's node: pass --gpu, or a node number" >&2; exit 1; }
    node_file="$sysfs/bus/pci/devices/${gpus[0]}/numa_node"
    [ -r "$node_file" ] || { echo "no $node_file: ${gpus[0]} is not a PCI device on this host" >&2; exit 1; }
    numa_node="$(cat "$node_file")"
    [ "$numa_node" != "-1" ] || { echo "${gpus[0]} reports no NUMA node (-1): pass --numa-node <n>, or omit it" >&2; exit 1; }
  fi
  [[ "$numa_node" =~ ^(0|[1-9][0-9]{0,2})$ ]] || { echo "--numa-node must be a node number or auto; got $numa_node" >&2; exit 1; }
  [ -d "$sysfs/devices/system/node/node$numa_node" ] || { echo "this host has no NUMA node $numa_node" >&2; exit 1; }
  wrapper=(numactl "--cpunodebind=$numa_node" "--membind=$numa_node")
fi

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
[ "$num_volumes" -ge 1 ] || { echo "$shape declares no verity volume, but the worker image disk is one" >&2; exit 1; }
[ "$((${#weights[@]} + 1))" -eq "$num_volumes" ] \
  || { echo "$shape attaches $num_volumes verity volume(s), the worker image disk and $((num_volumes - 1)) weights disk(s); got ${#weights[@]} weights disk(s)" >&2; exit 1; }
[ "$num_nics" -eq 1 ] || { echo "launch-td.sh emits exactly one NIC; $shape declares $num_nics" >&2; exit 1; }
[ "${#nvswitches[@]}" -eq "$num_nvswitches" ] || { echo "$shape needs $num_nvswitches NVSwitch(es); got ${#nvswitches[@]}" >&2; exit 1; }
[ "$hugepages" -eq 0 ] || { echo "hugepages change the QEMU NUMA layout; not supported by this launcher" >&2; exit 1; }
for file in ovmf.fd bzImage initramfs.cpio.gz rootfs.img.verity metadata.json; do
  [ -f "$release/$file" ] || { echo "$release has no $file" >&2; exit 1; }
done
for suffix in img.verity roothash size digest; do
  [ -f "$image_prefix.$suffix" ] || { echo "$image_prefix.$suffix is missing: pack the worker image disk with pack-image.sh" >&2; exit 1; }
done
cmdline="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["cmdline"])' "$release/metadata.json")"

mkdir -p "$state"
[ -f "$state/data.img" ] || truncate -s 64M "$state/data.img"  # present because dstack-vmm always attaches one; kuno-app never mounts it
printf '%s %s %s\n' "$(cat "$image_prefix.roothash")" "$(cat "$image_prefix.size")" "$(cat "$image_prefix.digest")" > "$state/image.txt"
: > "$state/weights.txt"

args=(-accel kvm -cpu host -nographic -nodefaults
  -chardev "stdio,id=com0,logfile=$state/serial.log,logappend=on" -serial chardev:com0
  -bios "$release/ovmf.fd" -kernel "$release/bzImage" -initrd "$release/initramfs.cpio.gz")
[ "$hotplug_off" -eq 1 ] && args+=(-global ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off)
[ "$hole64" -gt 0 ] && args+=(-global "q35-pcihost.pci-hole64-size=$(printf '0x%x' "$hole64")")
args+=(-drive "file=$release/rootfs.img.verity,if=none,id=hd0,format=raw,readonly=on" -device "virtio-blk-pci,drive=hd0")
args+=(-drive "file=$state/data.img,if=none,id=hd1,format=raw" -device "virtio-blk-pci,drive=hd1")
args+=(-drive "file=$image_prefix.img.verity,if=none,id=vol0,format=raw,readonly=on" -device "virtio-blk-pci,drive=vol0,serial=kuno-image")
index=1
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
args+=(-device "vhost-vsock-pci,guest-cid=$cid")
args+=(-machine "q35,kernel-irqchip=split,confidential-guest-support=tdx,hpet=off")
args+=(-object "{\"qom-type\":\"tdx-guest\",\"id\":\"tdx\",\"quote-generation-socket\":{\"type\":\"vsock\",\"cid\":\"2\",\"port\":\"$qgs_port\"}}")
if [ "$num_gpus" -gt 0 ]; then
  args+=(-object "iommufd,id=iommufd0")
  slot=1
  for device in "${gpus[@]}" "${nvswitches[@]}"; do
    args+=(-device "pcie-root-port,id=pci.$slot,bus=pcie.0,chassis=$slot" -device "vfio-pci,host=$device,bus=pci.$slot,iommufd=iommufd0")
    slot=$((slot + 1))
  done
fi
args+=(-smp "$cpus" -m "${memory_mib}M")
args+=(-fw_cfg "name=opt/kuno/image,file=$state/image.txt")
args+=(-fw_cfg "name=opt/kuno/weights,file=$state/weights.txt")
[ -n "$env_file" ] && args+=(-fw_cfg "name=opt/kuno/env,file=$env_file")
[ -n "$seed_file" ] && args+=(-fw_cfg "name=opt/kuno/hotkey.seed,file=$seed_file")
args+=(-append "$cmdline")

if [ "$run" = 1 ]; then
  exec "${wrapper[@]}" "$qemu" "${args[@]}"
fi
printf '%q ' "${wrapper[@]}" "$qemu" "${args[@]}"
printf '\n'
