#!/usr/bin/env python3
"""Inside a container on a smci355 node: does HIP device i sit behind the same PCIe switch as ionic_i?

io_inject.py pairs GPU i with ionic_i, and KFD lists the GPUs in the NICs' order on these nodes; this prints the
check per device (HIP index, its PCI address, ionic_i's, OK / MISMATCH).
"""
import os

import torch


def switch_of(bdf):
    parts = os.path.realpath(f"/sys/bus/pci/devices/{bdf}").split("/")
    return "/".join(parts[3:6])


bad = 0
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    bdf = f"{getattr(p, 'pci_domain_id', 0):04x}:{p.pci_bus_id:02x}:{getattr(p, 'pci_device_id', 0):02x}.0"
    nic = os.path.basename(os.path.realpath(f"/sys/class/infiniband/ionic_{i}/device"))
    ok = switch_of(bdf) == switch_of(nic)
    bad += not ok
    print(f"hip{i} {bdf} ionic_{i} {nic} {'OK' if ok else 'MISMATCH'}")
print("TOPO_OK" if not bad else f"TOPO_MISMATCH {bad}")
