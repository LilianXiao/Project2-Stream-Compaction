CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Lilian Xiao
  * [LinkedIn](https://www.linkedin.com/in/lilian-xiao/), [personal website](https://lilianxiao.carrd.co/), [art website](https://lilianxvis.carrd.co/)
* Tested on: Windows 11, Intel(R) Core(TM) Ultra 9 185H (2.50 GHz), 16.0 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB)
Intel(R) Arc(TM) Graphics (128 MB) (Personal Laptop)

**Important: CmakeList changes**

Added the below to mitigate possible errors resulting from CUDA version.
```
if(MSVC)

    target_compile_options(stream_compaction PRIVATE 

        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"

    )

endif()
```

### CUDA GPU Stream Compaction

Stream compaction is a particularly useful parallel computing algorithm that can filter through extremely large arrays and datasets, such as removing undesired elements.  This project demonstrates multiple different implementations of stream compaction to remove all zeros from a large array populated entirely with integers.  The compaction algorithm is hand in hand with a prefix sum scanning algorithm, and has notably been implemented for CPU, naive and work-efficient GPU versions, and using the Thrust library exclusive scan.

# Performance Analysis

## Block size optimization
I conducted a sweep of work-efficient scanning and compaction GPU optimization.  I made block size into an external variable for the purposes of testing from `32` to `1024`, with a fixed array size of `1 << 24`.

**Work-Efficient Scan Performance by Block Size**

```
Block   32: median 5.7387 ms, min 5.6106 ms
Block   64: median 5.6935 ms, min 5.6726 ms
Block  128: median 5.6591 ms, min 5.6244 ms
Block  256: median 5.6514 ms, min 5.6221 ms
Block  512: median 5.6108 ms, min 5.5902 ms
Block 1024: median 5.6367 ms, min 5.5957 ms
```

**Work-Efficient Compaction Performance by Block Size**

```
Block   32: median 7.3318 ms, min 7.2602 ms
Block   64: median 7.2223 ms, min 7.1946 ms
Block  128: median 7.2182 ms, min 7.1946 ms
Block  256: median 7.2038 ms, min 7.1690 ms
Block  512: median 7.1803 ms, min 7.1557 ms
Block 1024: median 7.1690 ms, min 7.1270 ms
```

It seems that scanning performance is relatively unaffected by changing the block size, but there is a slight observable improvement in performance for the compaction algorithm.  This could potentially indicate that larger block sizes result in fewer total launches (since I ended up trying to make a part 5 optimization where top layers will get lumped together).  Compaction in general performs slightly slower than scanning, likely due to the mapping and scattering.

## Comparing Across All Implementations: GPU vs. CPU based on array size

<img width="600" height="371" alt="CPUvsGPU_Scan" src="https://github.com/user-attachments/assets/42603c9b-fffd-48ae-a9dc-36c8539ded9d" />

Array size evaluation began from `1<<8` to `1<<24`.  For sizes `1<<8` and `1<<10`, the CPU implementation was much faster than the GPU scans.  However, at `1<<20`, the gap begins to close, and for sizes `1<<22` and greater, the work-efficient GPU scan and Thrust scan began visibly outperforming both the CPU scan and naive GPU scan.

* The CPU is able to handle arrays at small sizes much faster than the GPU in this case.  It seems that for these cases, GPU operations actually put on more fixed costs, and results in slower performance relatively speaking.
* Eventually, once the array size has significantly increased, the CPU has to do a lot of unavoidable work and becomes a lot slower compared to the capabilities of GPU parallel computing.
* Naive scans are generally not great, doing around `nlog(n)` work for `log(n)` launches.  This explains the great disparity between the naive and work-efficient implementations.
* Although the work-efficient scan tries its best, Thrust is just too speedy.

## Trying to decipher Thrust using Nsight


**Test Output for Block Size 128, Array Size `1<<24`**

```
****************
** SCAN TESTS **
****************
    [  20   3  37  12  22   1   0   7  14  43  29  35  20 ...  20   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 9.0276ms    (std::chrono Measured)
    [   0  20  23  60  72  94  95  95 102 116 159 188 223 ... 410880383 410880403 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 9.2979ms    (std::chrono Measured)
    [   0  20  23  60  72  94  95  95 102 116 159 188 223 ... 410880341 410880352 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 15.4319ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 15.3708ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 5.6281ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 5.68755ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.29434ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.2073ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   1   2   0   1   0   1   0   3   3   3   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 32.9445ms    (std::chrono Measured)
    [   1   1   2   1   1   3   3   3   2   1   2   3   2 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 31.8535ms    (std::chrono Measured)
    [   1   1   2   1   1   3   3   3   2   1   2   3   2 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 48.9128ms    (std::chrono Measured)
    [   1   1   2   1   1   3   3   3   2   1   2   3   2 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 7.87866ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 7.9104ms    (CUDA Measured)
    passed
```
