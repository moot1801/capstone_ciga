from pathlib import Path
import shutil

def print_to(file, *args, sep=" ", end="\n"):
    file_path = Path("/home/jinholee/Ciga/logs/",file)

    with open(file_path, "a", encoding="utf-8") as f:
        print(*args, sep=sep, end=end, file=f)  

def nuke_dir(recreate: bool = True):
    d = Path("/home/jinholee/Ciga/logs/")
    if d.exists():
        shutil.rmtree(d)
    if recreate:
        d.mkdir(parents=True, exist_ok=True)        

def log_weight_stats(module, file_path="/home/jinholee/Ciga/logs/mlp_weight_stats.txt"):
    with open(file_path, "a", encoding="utf-8") as f:
        for name, p in module.named_parameters():
            t = p.detach()
            f.write(
                f"{name:20s} "
                f"shape={tuple(t.shape)} "
                f"min={t.min().item():.4e} "
                f"max={t.max().item():.4e} "
                f"mean={t.mean().item():.4e} "
                f"std={t.std(unbiased=False).item():.4e}\n"
            )
        f.write("\n\n")