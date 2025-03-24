#!/usr/bin/env bash

MODEL_PATH="/home/niall/try2/model/Model-8.0B-BF16.gguf"
PROMPT="What is 1+1"
SYS_PROMPT="Remember to use <reasoning> </reasoning> to delimit your chain of thought, and also to use <answer> </answer> to delimit the final answer"
OUTFILE="gpu_bench.csv"
COMMON_OPTS=(
  -m "$MODEL_PATH"
  -p "$PROMPT"
  -sys "$SYS_PROMPT"
  -st
  -c 1024
  -t 16
  --temp 0.1
  --color
  --prio 3
  --flash-attn
)

echo "n_gpu_layers,tokens_per_sec" > "$OUTFILE"

for n in $(seq 1 33); do
    echo -n "Offloading $n layer(s)... "

    output=$(./build/bin/llama-cli \
        "${COMMON_OPTS[@]}" \
        --n-gpu-layers "$n" 2>&1)
    echo
    echo $output  > single-run.txt
    echo
    tps=$(echo "$output" |sed 's/context_print/\n\n/g' |awk -F',' '{print $2}' |awk '{print $1}' |tac |grep -m1 -v '^[[:space:]]*$')
    #tps=$(echo $output |grep 'tokens per second' | awk -F',' '{print $2}' | awk '{print $1}')
    #tps=$(echo "$output" | grep 'tokens per second' | awk '{print $(NF)}')

    if [[ -n "$tps" ]]; then
        echo "$n,$tps" >> "$OUTFILE"
        echo "$tps tok/s"
    else
        echo "$n,FAIL" >> "$OUTFILE"
        echo "failed"
    fi
done
