# Sourced inside the containers. On every smci355 node HIP device i and ionic_i sit behind the same PCIe switch, but
# the nodes enumerate in two orders, so device i is on a different backend rail (benic<r>) depending on the node:
#   n08-21 / n08-29 / n09-21 / n09-29: devices 0..7 on rails 1 2 4 3 5 6 8 7
#   n08-25 / n08-33:                   devices 0..7 on rails 3 1 4 2 7 5 8 6
# MoRI pairs local rank i with local rank i of the other nodes (its "rails" are local ranks) and io_inject.py pairs
# device i with device i, so between the two kinds of node 6 of the 8 pairs cross rails. rail_perm prints the HIP
# device order that puts local rank i on rail CANON[i] (the first order) on this node; RAIL_ALIGN=1 exports it as
# HIP_VISIBLE_DEVICES so that every pair stays on one rail.
rail_perm() {
    local canon=(1 2 4 3 5 6 8 7) r=() i j out=""
    for i in 0 1 2 3 4 5 6 7; do
        r[$i]=$(ls /sys/class/infiniband/ionic_$i/device/net/ 2> /dev/null | sed 's/^benic//; s/p1$//')
    done
    for i in 0 1 2 3 4 5 6 7; do
        for j in 0 1 2 3 4 5 6 7; do
            [ "${r[$j]}" = "${canon[$i]}" ] && { out="$out${out:+,}$j"; break; }
        done
    done
    [ "$(echo "$out" | tr ',' '\n' | sort -u | wc -l)" = 8 ] && echo "$out" || echo "0,1,2,3,4,5,6,7"
}
rail_align() {
    if [ "${RAIL_ALIGN:-0}" = 1 ]; then
        local p; p=$(rail_perm)
        [ "$p" = "0,1,2,3,4,5,6,7" ] || export HIP_VISIBLE_DEVICES=$p
    fi
}
