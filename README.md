# amrex_profiling_analyzer

This tools aims to complement AMReX's base profilng and profVis in the following aspects:
* provide easy access to profiling results
* analyze per-process time cost along critical call paths
* find where the most expensive functions are called
* detail the high-level expensive functions including what child functions it calls and their costs

## Dependency
This tool is based on AMReX's base profiling. To enable base profiling, one needs to add `-DAMReX_BASE_PROFILING` and `-DAMReX_TRACE_PROFILING` while building AMReX.

The other dependent python modules (numpy, matpyplotlib, etc) can be easily installed ad-hoc. 

## Usage
The exemplary usages are illusrated in `analyze.py`. 
