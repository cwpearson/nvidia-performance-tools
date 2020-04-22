# Coalescing Examples

Examples for measuring memory coalescing in Nsight Compute

To really see this in Nsight Compute, you need at least 2019.5.
Earlier versions do not show requested command line profiler metrics.


From the Programming guide section 5
Device memory can be accessed in 32B, 64B or 128B transactions.


CC3.x: 
Global accesses are cached in L2, and may be cached in L1 if read-only.

It appears that memory requests are 128B.

Cache line is 128 bytes
If in L1 and L2, served with 128 byte transactions
If in L2 only, 32-byte transactions

Each memory request is broken down into cache line requests that are issued independently. So, a 128B request may turn into some number of 128B or 32B cache line requests / transactions.

CC5.x
 Always cached in L2, which is same as CC3.x

CC6.x same as CC5.x
CC7.x same as CC5.x

*sector*: an L1 line

## Measuring Memory Coalescing with Nsight Compute

Coalescing means we want to minimize the number of memory or cache transactions that are needed to serve a request.

There are four `indirect` kernels executed each iteration of this code.
* float, coalesced
* float, uncoalesced
* double, coalesced
* double, uncoalesced

Each thread increments one element in global memory.
The access is indirect through an integer array, which is permuted on the host to control coalescing.

**On Pascal (GTX 1070, GP102 or 104 or something (not GP100) CC6.1)**

On GP 104, Caches global loads in L2 only (Pascal Tuning Guide 1.4.3.2)

On pascal, this code makes `LDG.E` accesses, and L2 accesses are 32B.

10,000 floats -> 40KB
40 KB into 32B / request = 1250 requests.
if coalesced, each request is a single "sector" (?), so also 1250 sectors
If uncoalesced, each 32B request may be 8 separate 4B floats each from a different sector, so up to 10K sectors.


We can look at the "Source" page for the `ld.global.f32` instruction:

| | coalesced | uncoalesced |
|-|-|-|
| Instructions Executed | 313 | 313 |
| Predicated On-Thread Instructions | 10000 | 10000 |
| Memory Access Size | 32 | 32 |
| Memory L2 Transactions Global | 1250 | 9874 |

Memory access size means a 32-bit load instead of 64-bit.
10000 floats into 32 threads/warp is 313 instructions.
If each of those 313 instructions was 32 loads, each would be a minimum of 4 32B L2 transactions, or 1252 transactions.
We can tell some of those 313 instructions are not full, since 313 * 32 = 10,016, and we only had 10,000 instructions predicated-on (thus, 1250).

In any case, we expect ~1250 transactions, which we observe for coalesced.
For uncoalesced, we see 9874, or about 8x too many.
So each 32 loads is ~32 L2 transactions, or fully uncoalesced.

A similar analysis with `double`s (`ld.global.f64`) instead of floats: 

| | coalesced | uncoalesced |
|-|-|-|
| Instructions Executed | 313 | 313 |
| Predicated On-Thread Instructions | 10000 | 10000 |
| Memory Access Size | 64 | 64 |
| Memory L2 Transactions Global | 2500 | 9950 |

Each load instruction is now 32 8B loads, or a minimum of 8 32B L2 transactions.
313 * 8 transactions = 2504 transactions
Instead we observe nearly 4x the number of transactions.
The uncoalescing is 1/2 as bad here (4x instead of 8x) because each 32B L2 transaction can only handle 4 doubles instead of 8 floats.

We can also consider the kernel as a whole:

Coalesced accesses
```
nv-nsight-cu-cli --kernel-id "::indirect:1" --metrics tex__texin_requests_global_ld_uncached,lts__request_tex_read_sectors_global_ld_uncached,lts__request_tex_read_bytes_global_ld_uncached main
```

```
[24874] main@127.0.0.1
  indirect, 2020-Apr-22 09:04:29, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    lts__request_tex_read_bytes_global_ld_uncached                                   Kbyte                             80
    lts__request_tex_read_sectors_global_ld_uncached                                sector                          2,500
    tex__texin_requests_global_ld_uncached                                                                          2,500
    ---------------------------------------------------------------------- --------------- ------------------------------
```

Uncoalesced Accesses
```
nv-nsight-cu-cli --kernel-id "::indirect:2" --metrics tex__texin_requests_global_ld_uncached,lts__request_tex_read_sectors_global_ld_uncached,lts__request_tex_read_bytes_global_ld_uncached main
```

```
[24760] main@127.0.0.1
  indirect, 2020-Apr-22 09:03:40, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    lts__request_tex_read_bytes_global_ld_uncached                                   Kbyte                         357.02
    lts__request_tex_read_sectors_global_ld_uncached                                sector                         11,157
    tex__texin_requests_global_ld_uncached                                                                          2,500
    ---------------------------------------------------------------------- --------------- ------------------------------
```

If coalesed, 1250 + 1250 = 2500 32B requests
1250 + 1250 = 2500 sectors.

If not, 1250 + 1250 = 2500 32B requests
1250 + 10K = max of 11250 sectors.


**On Volta (Titan V, CC7.0)**

On Volta, global memory loads are cached in L1.

On Volta, this code makes `LDG.E.SYS` accesses, so we get 128B L1 requests and 32B L2 requests.

Alternatively, we can look at the "Source" page for the `ld.global.f32` instruction:

| | coalesced | uncoalesced |
|-|-|-|
| Instructions Executed | 313 | 313 |
| Memory Access Size | 32 | 32 |
| Memory L1 Transactions Global | 313 | 9515 |
| Memory L2 Tranactions Global | 1250 | 9885 |

Similarly, 313 instructions.
When coalesced, this results in 313 L1 requests.
If uncoalesced, each 128B instruction may ultimately be 32 separate 4B floats each from a different sector, so up to 10K sectors / transactions.

When coalesced, this is ~313 * 4 = 1250 L2 requests.
If uncoalesced, each 128B instruction may ultimately be 32 separate 4B floats each from a different sector, so up to 10K sectors / transactions.

We can also consider the `ld.global.f64` version:

| | coalesced | uncoalesced |
|-|-|-|
| Instructions Executed | 313 | 313 |
| Memory Access Size | 64 | 64 |
| Memory L1 Transactions Global | 625 | 9740 |
| Memory L2 Tranactions Global | 2500 | 9947 |

Now, each instruction (32 8B loads) generates two 128B L1 transactions (coalesced), or 32 128B L1 transactions (uncoalesced).

Now, each instruction (32 8B loads) generates 8 32B L2 transactions (coalesced), or 32 128B L2 transactions (uncoalesced).

We can also consider the entire kernel.

The uncoalesced results:
```
nv-nsight-cu-cli --kernel-id "::indirect:1" --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum main
```

```
[362] main@127.0.0.1
  indirect(float*,int*,unsigned long), 2020-Apr-22 14:25:36, Context 1, Stream 7
    Section: Command line profiler metrics
---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                            626
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                           2500
    ---------------------------------------------------------------------- --------------- ------------------------------
```

The coalesced results:
```
nv-nsight-cu-cli --kernel-id "::indirect:2" --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum main
```

```
[387] main@127.0.0.1
  indirect(float*,int*,unsigned long), 2020-Apr-22 14:25:36, Context 1, Stream 7
    Section: Command line profiler metrics
---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                                request                            626
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                          10093
    ---------------------------------------------------------------------- --------------- ------------------------------
```




## Instruction Profiling

Compare `Instructions Executed`, `L1 Transactions`, and `L2 Transactions`.

## Resources

This content is inspired by https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/

Metrics Names: https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_derived_metrics_api

