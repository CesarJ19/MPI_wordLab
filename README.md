# Lab 2: Parallel Word Counting in a Text Corpus with MPI

## 1. Title

Parallel word counting in a text corpus with MPI: static distribution by number of files versus static distribution by file size.

## 2. Problem Description

The dataset is generated with `generator.py`. It contains 3000 text files (`dataset/file_XXXX.txt`) and a query file, `dataset/consulta.txt`, with 200 words. The files have very different sizes (about 60% small, 30% medium and 10% large) and the corpus has 44,951,458 words in total.

The task is to count how many times each query word appears in the corpus and to report the 10 most frequent ones. The problem is solved with the sequential baseline provided by the instructor and with two MPI programs written with mpi4py. All of them must produce the same result, and the MPI programs are compared in terms of total time, speedup, efficiency and load imbalance.

Repository files:

- `generator.py`, `baseline_secuencial.py`, `run_all.sh`: provided, not modified.
- `mpi1.py`: MPI version 1.
- `mpi2.py`: MPI version 2.
- `README.md`: this report.

## 3. Environment and Execution Instructions

Everything runs inside the Docker image `augustosalazar/slim-mpi:2` (Python, mpi4py and Open MPI). The host is a laptop with an Intel Core i5-10300H at 2.50 GHz (4 cores, 8 logical processors) running Windows 11 Home and Docker Desktop (Docker 29.6.2). The commands were typed in PowerShell. Inside the container `nproc` reports 8 CPUs.

The number of processes was obtained with `detectar_procesos_mpi.py`, the script linked in the lab statement. It tested N from 1 to 8, with three measurements per N, and recommended N = 8 (median time 2.035 s, against 2.369 s for N = 4). Therefore p = 8 in all MPI experiments.

Commands, taken from the instructor's lab statement and Readme (PowerShell, run from the folder with the scripts). Generate the dataset:

```powershell
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py
```

Run the sequential baseline:

```powershell
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

Run all the experiments with the provided script:

```powershell
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

`run_all.sh` generates the dataset, runs the baseline, and runs `mpi1.py` and `mpi2.py` several times. For each MPI run it executes, inside the container, the following line, where `$NP` is the number of processes:

```sh
mpirun --allow-run-as-root --oversubscribe -np "$NP" python "$APP_DIR/mpi1.py"
```

The script reads the number of processes and runs from the variables `NP_LIST` (default `2 4 8`) and `RUNS` (default 3). In this lab `NP_LIST` was set to 8, passing the variable to the container, so only p = 8 was run, with 3 runs per version. On Windows, `run_all.sh` has CRLF line endings and fails in the container, so the carriage returns were removed in memory before giving the script to the shell. The file itself was not modified. The logs are written to `results/logs`.

### 3.a Experimental Plan

1. Generate the dataset (fixed seed 42).
2. Run the sequential baseline and record T_seq and its output.
3. Obtain p with `detectar_procesos_mpi.py`.
4. Run each MPI version three times with p processes, recording the total time and, for every process, the number of files and the local time.
5. Compute for each version T_p (mean of the total times), speedup S_p = T_seq / T_p, efficiency E_p = S_p / p and load imbalance LI = max(t_i) / mean(t_i). The relative spread (max(t_i) - min(t_i)) / mean(t_i) is also reported.
6. Compare both versions and answer the analysis questions.

### 3.b Sequential baseline

`baseline_secuencial.py` loads the query words into a set. Then it reads every `file_*.txt` line by line, splits each line into words, converts them to lower case and increments a counter when the word belongs to the query set. It prints the elapsed time, the number of tokens and occurrences, and the top 10, and it writes `dataset/baseline_results.csv`, which is used to check the MPI results.

### 3.c MPI version 1

`mpi1.py` follows the steps of the lab statement:

1. Rank 0 reads `consulta.txt`.
2. Rank 0 sends the query words to all processes with `comm.bcast`.
3. Rank 0 lists the `file_*.txt` files, sorted by name.
4. The list is split into p contiguous blocks with the same number of files (375 each for p = 8) and sent with `comm.scatter`. This is a static distribution.
5. Each process counts the query words in its own files and measures its local time with `MPI.Wtime()`.
6. `comm.gather` sends the partial counters, the number of files and the local time of each process to rank 0.
7. Rank 0 adds the partial counters and prints the files and local time of every process, the LI, the top 10, and whether the result equals `baseline_results.csv`. It also prints `EXECUTION_TIME`, which `run_all.sh` reads from the logs.

The total time starts after a `comm.Barrier()` and ends when rank 0 has the final result, so it includes the broadcast, the distribution and the gather.

### 3.d Test procedure

Correctness: the top 10 and the occurrence counts of both MPI programs must be equal to the baseline. Each program prints `MATCHES_BASELINE=True` when the complete table of counts is identical. Performance: each version is run three times with p = 8 on the same dataset, and T_p is the mean of the three total times. T_seq is taken from the baseline run executed inside `run_all.sh`, in the same session as the MPI runs.

## 4. Experimental Plan Execution

### 4.a Sequential baseline timing

T_seq = 24.134874 s. The baseline processed 3000 files and 44,951,458 tokens and found 3,631,778 occurrences. The top 10 is:

| Word | Count |
|---|---|
| a | 785774 |
| para | 392156 |
| sus | 228913 |
| otros | 105530 |
| ante | 99832 |
| unos | 88794 |
| otra | 83901 |
| vosotros | 61617 |
| mios | 58420 |
| tuya | 56635 |

This matches the example in the instructor's Readme, although the time is much higher than the roughly 5 s shown there.

### 4.b MPI version 1 timing results

| Run | Total time (s) |
|---|---|
| 1 | 6.338626 |
| 2 | 6.538507 |
| 3 | 6.450726 |

T_p = 6.4426 s, S_p = 24.1349 / 6.4426 = 3.746 and E_p = 3.746 / 8 = 0.468. All runs printed `MATCHES_BASELINE=True`.

### 4.c Load imbalance evidence (version 1)

Local processing time of each process, in seconds. Every process had 375 files.

| Run | rank 0 | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 | rank 6 | rank 7 | LI | Spread |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.187 | 5.321 | 6.307 | 6.235 | 5.698 | 5.834 | 5.885 | 6.050 | 1.0618 | 0.1660 |
| 2 | 6.353 | 5.504 | 6.503 | 6.404 | 5.957 | 6.118 | 6.110 | 6.282 | 1.0567 | 0.1622 |
| 3 | 6.368 | 5.452 | 6.419 | 6.309 | 5.765 | 5.951 | 5.917 | 6.146 | 1.0626 | 0.1601 |

The mean LI is 1.0604 and the mean spread is 0.1628. The processes read different amounts of text: rank 1 processed 4.80 million tokens and rank 2 processed 6.18 million, which matches the fastest and slowest local times.

### 4.d MPI version 2: how it corrects the imbalance, and its results

Version 1 gives every process the same number of files, but the files have different sizes, so the work is not equal. `mpi2.py` has the same structure as version 1 (broadcast of the query, scatter of the files, local count, gather of the results) and only changes how the files are assigned. The size of each file in bytes is used as an estimate of its cost, since the counting time is roughly proportional to the number of words. The sizes are read in parallel, each process reading a slice of the file list, and rank 0 gathers them. Then the files are sorted from largest to smallest and each one is given to the process with the smallest accumulated size (greedy LPT heuristic). The assignment is still static, since it is decided before the counting starts, and it is sent with `comm.scatter`. The program also prints `PARTITION_TIME`, the time spent deciding and distributing the assignment.

| Run | Total time (s) | LI | Spread | Partition time (s) |
|---|---|---|---|---|
| 1 | 6.652435 | 1.0128 | 0.0204 | 0.702221 |
| 2 | 6.711831 | 1.0080 | 0.0130 | 0.615271 |
| 3 | 6.749043 | 1.0090 | 0.0150 | 0.615112 |

In all runs every process read about 5.62 million tokens (374 to 376 files) and the local times of the eight processes differ by less than 0.15 s. All runs printed `MATCHES_BASELINE=True`.

Comparison (p = 8, mean of 3 runs):

| Version | T_p (s) | S_p | E_p | LI | Largest local time (s) |
|---|---|---|---|---|---|
| Sequential | 24.1349 | 1.000 | 1.000 | not applicable | not applicable |
| MPI v1 | 6.4426 | 3.746 | 0.468 | 1.0604 | 6.409 |
| MPI v2 | 6.7044 | 3.600 | 0.450 | 1.0099 | 6.054 |

The ideal values for p = 8 are S_p = 8, E_p = 1 and a time of about 3.02 s.

## 5. Analysis

### 5.a Parallel performance

Version 1 improved the sequential baseline: the time went from 24.13 s to 6.44 s, a speedup of 3.75. This is far from the ideal speedup of 8, with an efficiency of 0.47. Version 2 reached 3.60.

The gap is not explained by communication, because in version 1 the difference between the total time and the largest local time is only about 0.03 s. It is not explained mainly by imbalance either: with LI = 1.06 the slowest process takes about 6% more than the average. The main cause is that each process runs slower when the eight run together. The eight local times of run 1 add up to about 47.5 s, while the sequential program does the whole job in 24.1 s, so each unit of work took about twice as long. The laptop has 4 physical cores and 8 logical processors, so two processes share each core, and all of them read their files through the same mounted folder. The measurements of `detectar_procesos_mpi.py` agree with this, since going from 4 to 8 processes only reduced the time from 2.37 s to 2.04 s. How much of the slowdown comes from shared cores and how much from file access was not measured.

### 5.b Load balance

Version 1 showed clear evidence of imbalance. Although every process had 375 files, the local times ranged from 5.32 s (rank 1) to 6.31 s (rank 2) in run 1, with similar patterns in the other runs. The LI was between 1.057 and 1.063 and the spread was about 0.16. Having the same number of files does not mean having the same work when file sizes vary.

Version 2 reduced the imbalance: the LI dropped to between 1.008 and 1.013 and the spread to between 0.013 and 0.020. The largest local time fell from 6.409 s to 6.054 s on average, a saving of 0.355 s. However, this did not produce a real improvement. Deciding the partition costs 0.644 s on average, more than the time saved, and the total time of version 2 (6.704 s) ended 0.262 s (about 4%) above that of version 1 (6.443 s). Since the imbalance of version 1 is only about 6%, perfect balancing could not save more than that, which was not enough to pay for the extra step in this environment.

### 5.c Limitations

- All measurements come from one Windows laptop running Docker, with the dataset in a folder mounted from the host. This probably slows down file access, and the baseline (24 s) is much slower than the example of the instructor (about 5 s). The absolute times cannot be transferred to other machines.
- The laptop has 4 physical cores, so p = 8 does not mean 8 independent cores, and the computer was also running Windows and other programs. This explains part of the gap to the ideal speedup, but its share was not isolated.
- Only three runs per version were made. The total times vary by about 0.1 s between runs of the same version, and the difference between versions (0.26 s) is only two to three times larger. In one extra run outside the official ones, version 2 was 0.12 s faster than version 1, the opposite of the official result. The order of the two versions is therefore not firmly established.
- A single dataset, a single value of p and a single baseline measurement were used, and the file system cache was not controlled. The size in bytes is only an estimate of the cost of a file.

These limitations weaken mostly the comparison between the two MPI versions, because the difference is small. They weaken much less the conclusion that both versions are about 3.6 to 3.7 times faster than the baseline, since that gap is far larger than the variation between runs.

## 6. Conclusions

For this problem and environment, the static distribution by number of files (version 1) is recommended. It had the best total time (T_p = 6.44 s, speedup 3.75, efficiency 0.47), against 6.70 s, 3.60 and 0.45 for version 2, even though its balance is worse (LI 1.06 against 1.01). The imbalance of version 1 is small, so balancing the load saves at most about 0.36 s, while the size-based partition of version 2 costs about 0.64 s. Version 2 would be more convenient if the imbalance were larger, if the work per file were more expensive compared with reading the file sizes, or if the sizes were cheaper to obtain. Because the difference between versions is about 4% and was measured with three runs on one machine, the recommendation is moderate. The clear result is that both MPI versions are about 3.6 to 3.7 times faster than the sequential program, far from the ideal speedup of 8.
