#!/usr/bin/env bash
#
#for this to work you need
#   a runnung ollama server that stores its models in /usr/share/ollama/... 
#   you should already have pulled a copy of the model you are finetuning 
#   then just set og_model to its name.


set -e

og_model=qwen2.5:1.5b-instruct

wd=$(pwd)
gguf_path=$wd/model/unsloth.F16.gguf

ollama show --modelfile $og_model | sed -E 's|/usr/share/ollama/.ollama/models/blobs/sha256-[0-9a-f]+|'$(echo -n $gguf_path)'|g' > Modelfile.tmp
ollama create unsloth -f Modelfile.tmp
rm Modelfile.tmp

ollama run unsloth
