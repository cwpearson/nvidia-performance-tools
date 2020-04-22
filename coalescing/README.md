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

In this code, we load and store one 4B float per thread.
To determine which float to load, we also load one 4B int per thread.

Each memory request can contain 32B (source?).


**On Pascal (GTX 1070, CC6.1)**

On pascal, this code makes uncached `LDG.E` accesses, so there are 32B requests (only cached in L2).



**For the indices**

10,000 ints -> 40KB
40 KB into 32B / request = 1250 requests.
Each request should be satisfied by a single "sector" (?), so 1250 sectors.
This is because the accesses into the offset array is coalesced

**For the floats**

10,000 floats -> 40KB
40 KB into 32B / request = 1250 requests.
if coalesced, each request is a single "sector" (?), so also 1250 sectors
If uncoalesced, each 32B request may be 8 separate 4B floats each from a different sector, so up to 10K sectors.

**Putting it Together**

If coalesed, 1250 + 1250 = 2500 32B requests
1250 + 1250 = 2500 sectors.

If not, 1250 + 1250 = 2500 32B requests
1250 + 10K = max of 11250 sectors.

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

**On Volta (Titan V, CC7.0)**

On Volta, this code makes cached `LDG.E.SYS` accesses, so we get 128B requests.

Why the cache behavior is different is unclear

**For the indices**

10,000 ints -> 40KB
40 KB into 128B / request = 313 requests (312.5)
Each request should be satisfied by 4 32B sectors except for the last request, which is 2 32B sectors, so 1250 sectors.
This is because the accesses into the offset array is coalesced.

**For the floats**

10,000 floats -> 40KB
40 KB into 128B / request = 313 requests (312.5)
if coalesced, same as above, for 1250 sectors.
If uncoalesced, each 128B request may ultimately be 32 separate 4B floats each from a different sector, so up to 10K sectors.

**Putting it Together**

If coalesed, 313 + 313 = 626 128B requests
1250 + 1250 = 2500 sectors.

If not, 313 + 313 = 626 128B requests
1250 + 10K = max of 11250 sectors.

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

## Resources

This content is inspired by https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/

Metrics Names: https://docs.nvidia.com/cupti/Cupti/r_main.html#r_host_derived_metrics_api

