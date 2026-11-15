# MPI version 1 - static file distribution (contiguous blocks).

import os
from collections import Counter

from mpi4py import MPI

CASE_SENSITIVE = False
TOP_N = 10


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not CASE_SENSITIVE:
        words = [w.lower() for w in words]
    return set(words)


def list_files(dataset_dir):
    return sorted(
        os.path.join(dataset_dir, n)
        for n in os.listdir(dataset_dir)
        if n.startswith("file_") and n.endswith(".txt")
    )


def split_contiguous(items, parts):
    """Static distribution: `parts` contiguous blocks of (almost) equal length."""
    base, extra = divmod(len(items), parts)
    blocks, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        blocks.append(items[start:end])
        start = end
    return blocks


def count_files(files, targets):
    counts, tokens = Counter(), 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                if not CASE_SENSITIVE:
                    words = [w.lower() for w in words]
                tokens += len(words)
                for w in words:
                    if w in targets:
                        counts[w] += 1
    return counts, tokens


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    comm.Barrier()                       # start all processes together
    t_start = MPI.Wtime()

    # rank 0 reads the query and broadcasts it
    query = None
    if rank == 0:
        query = load_query(os.path.join(dataset_dir, "consulta.txt"))
    query = comm.bcast(query, root=0)

    # rank 0 lists the files and distributes them statically
    blocks = None
    if rank == 0:
        blocks = split_contiguous(list_files(dataset_dir), size)
    my_files = comm.scatter(blocks, root=0)

    # local counting (timed separately)
    t0 = MPI.Wtime()
    local_counts, local_tokens = count_files(my_files, query)
    local_time = MPI.Wtime() - t0

    # gather partial results in rank 0
    info = (rank, len(my_files), local_time, local_tokens, dict(local_counts))
    gathered = comm.gather(info, root=0)

    # rank 0 builds the global result
    if rank == 0:
        global_counts, total_tokens = Counter(), 0
        for _, _, _, tokens, partial in gathered:
            global_counts.update(partial)
            total_tokens += tokens
        total_time = MPI.Wtime() - t_start

        times = [g[2] for g in gathered]
        mean_t = sum(times) / len(times)
        li = max(times) / mean_t if mean_t > 0 else float("nan")
        spread = (max(times) - min(times)) / mean_t if mean_t > 0 else float("nan")

        print("MPI version 1 (static contiguous distribution)")
        print(f"Processes (p): {size}")
        print("\nPer-process report:")
        for r, n_files, t, tokens, _ in sorted(gathered):
            print(f"  rank {r}: files={n_files}  tokens={tokens}  local_time={t:.6f} s")
        print(f"\nLOAD_IMBALANCE_LI={li:.4f}")
        print(f"RELATIVE_SPREAD={spread:.4f}")
        print(f"Total tokens read: {total_tokens}")
        print(f"Total occurrences found: {sum(global_counts.values())}")

        print(f"\nTop {TOP_N} query words in the corpus:")
        for word, cnt in global_counts.most_common(TOP_N):
            print(f"  {word}: {cnt}")

        # correctness check against the sequential baseline CSV
        csv_path = os.path.join(dataset_dir, "baseline_results.csv")
        if os.path.isfile(csv_path):
            baseline = {}
            with open(csv_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    w, c = line.rstrip("\n").split(",")
                    baseline[w] = int(c)
            print(f"MATCHES_BASELINE={baseline == dict(global_counts)}")
        else:
            print("MATCHES_BASELINE=unknown (run baseline_secuencial.py first)")

        print(f"EXECUTION_TIME={total_time:.6f}")


if __name__ == "__main__":
    main()
