#!/bin/bash

echo "Id,Transcription" > $2
python3 -m bin.infer --config_path="infer.yml" --model_dir="$1" | pv -lF "Total lines written: %b" -D 5 >> $2
