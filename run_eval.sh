#!/bin/bash
for f in ./abego_de_novo_fd_frag/*/; do
    python evaluate_design_set.py $f
done;
