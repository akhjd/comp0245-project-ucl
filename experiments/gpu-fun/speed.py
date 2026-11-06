"""Part 1 of the GPU experiments (6 Nov): how fast can the overnight base network train? Times training steps exactly
as train_base_torch.train does them (random batch from the whole dataset, conv ray filter + threat head, dropout 0.2,
AdamW, a loss.item() every step) on synthetic rows kept on the device, over network width, ray-filter detectors,
batch size and precision. Run in the GPU venv: <rocm python> -B runs/gpu-fun/speed.py OUT.json [cpu THREADS]"""
import json, sys, time
import torch, torch.optim, torch._dynamo      # before the stubs: torch looks up the real sklearn / scipy
sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-1000/-home-akhjd-ros2-ws-comp0245-project/6b3b2ff6-48e0-4813-9e6c-504ba49cfa02/scratchpad/speed")
import gpu_stubs                              # sklearn / scipy stand-ins (training never calls them)
sys.path.insert(0, "/home/akhjd/ros2_ws/comp0245_project")
import train_base_torch as tb

ROWS, RAYS, SCANS = 2_000_000, 144, 3        # about the combined run's training rows (r2: 1.83M)
N_IN = 11 + RAYS*SCANS


def time_config(dev, width, filters, batch, bf16, x, y, steps):
    tb.FILTERS = filters
    torch.manual_seed(0)
    net = tb.build_net(N_IN, width, SCANS, "conv", True, rays=RAYS, dropout=0.2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    params = sum(p.numel() for p in net.parameters())
    for i in range(steps + 10):
        if i == 10:
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
        idx = torch.randint(0, len(x), (batch,), device=dev)
        xb, yb = x[idx], y[idx]
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bf16):
            h = net.features(xb)
            loss = ((net.wheels(h).float() - yb)**2).mean() + \
                torch.nn.functional.binary_cross_entropy_with_logits(net.threat(h)[:, 0].float(), (yb[:, 0] > 0).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss.item()                            # train() reads the loss every step
    if dev == "cuda":
        torch.cuda.synchronize()
    per = (time.time() - t)/steps
    return {"device": dev, "width": width, "filters": list(filters), "batch": batch, "bf16": bf16, "parameters": params,
            "ms_per_batch": round(per*1000, 2), "rows_per_s": round(batch/per), "min_per_epoch": round(ROWS/batch*per/60, 2)}


if __name__ == "__main__":
    out = sys.argv[1]
    dev = "cpu" if len(sys.argv) > 2 and sys.argv[2] == "cpu" else "cuda"
    if dev == "cpu":
        torch.set_num_threads(int(sys.argv[3]))
    rows = ROWS if dev == "cuda" else 200_000
    x = torch.rand(rows, N_IN, device=dev)*10
    y = torch.randn(rows, 4, device=dev)
    grid = ([(w, f, b, p) for w in (128, 256, 512, 1024) for f in ((8, 4), (32, 16)) for b in (256, 1024, 4096, 8192)
             for p in (False, True)] if dev == "cuda" else [(128, (8, 4), b, False) for b in (256, 1024, 4096)])
    results = []
    for w, f, b, p in grid:
        steps = max(10, min(100, 400_000 // b))
        r = time_config(dev, w, f, b, p, x, y, steps)
        results.append(r)
        print(json.dumps(r), flush=True)
    meta = {"device_name": torch.cuda.get_device_name(0) if dev == "cuda" else f"cpu, {torch.get_num_threads()} threads",
            "torch": torch.__version__, "rows_for_epoch": ROWS, "note": "measured while the combined run used the CPU",
            "current_setting": "width 128, filters (8,4), batch 256, fp32"}
    json.dump({**meta, "results": results}, open(out, "w"), indent=1)
