"""NCCL all-reduce 带宽诊断：对比 intra-node(NVLink) vs inter-node(跨节点链路)。
torchrun 多机启动；NCCL_DEBUG=INFO 会在 stderr 打印实际传输层(Socket/IB/OFI)。
busbw = algbw * 2*(n-1)/n（ring all-reduce 标准公式）。"""
import os, time, socket
import torch, torch.distributed as dist


def bench(group, members, tag, rank):
    if group is None:
        return
    n = dist.get_world_size(group=group)
    is_member = rank in members
    if not is_member:
        return
    for mb in [4, 32, 128, 512]:
        numel = mb * 1024 * 1024 // 2          # bf16 = 2 bytes
        x = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):                      # warmup
            dist.all_reduce(x, group=group)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        iters = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x, group=group)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        algbw = (numel * 2) / dt / 1e9          # GB/s
        busbw = algbw * 2 * (n - 1) / n
        if rank == members[0]:
            print(f"[diag] {tag:24s} n={n:2d} size={mb:4d}MB  t={dt*1e3:7.2f}ms  "
                  f"algbw={algbw:6.1f}GB/s  busbw={busbw:6.1f}GB/s", flush=True)


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    if rank == 0:
        try:
            nccl_v = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:
            nccl_v = "?"
        print(f"[diag] world={world} torch={torch.__version__} nccl={nccl_v} host={socket.gethostname()}", flush=True)
    # 所有 rank 都要调 new_group（集合操作），仅成员参与 bench
    g_intra = dist.new_group(ranks=list(range(min(8, world))))
    g_inter = dist.new_group(ranks=[0, 8]) if world >= 16 else None
    bench(dist.group.WORLD, list(range(world)), "WORLD_allnodes", rank)        # 跨节点（受慢链路限）
    bench(g_intra, list(range(min(8, world))), "INTRA_node0_NVLink", rank)     # 单节点 NVLink
    if g_inter is not None:
        bench(g_inter, [0, 8], "INTER_2rank_crossnode", rank)                  # 纯跨节点单链路
    dist.barrier()
    if rank == 0:
        print("[diag] DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
