#! /bin/bash

ulimit -Sv 2000000
parallel -j-4 --eta 'python testRockSample.py {1}' ::: 1 2 3 4
