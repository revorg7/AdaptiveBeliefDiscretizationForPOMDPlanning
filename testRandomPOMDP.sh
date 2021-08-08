#! /bin/bash

ulimit -Sv 2000000
parallel -j-4 --eta 'python SamplingSearchPOMDP.py {1} {2} {3}' ::: 30 60 100 ::: 4 ::: 0.3 0.6 0.9
