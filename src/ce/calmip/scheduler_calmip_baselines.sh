#!/bin/bash

parent_dir=$(dirname "$(pwd)")

for file in "$parent_dir"/*ppo.py; do
    if [ -f "$file" ]; then
        cmd="bash scheduler_calmip_per_algo.sh ../$(basename "$file")"
        echo "Running: $cmd"
        eval $cmd
    fi
done
