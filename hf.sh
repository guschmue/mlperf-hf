#!/bin/bash

set -x

opt="--mlperf-config hf.conf --output results --name test --qps 20"
ort="--backend onnxruntime"
pt="--backend pytorch"

single="--scenario SingleStream"
server="--scenario Server --find-peak-performance --batchsize 16"
offline="--scenario Offline --batchsize 16"

declare -A singlestream_extras=(["bert-large-uncased:questionanswering"]="--batchsize 8")
declare -A offline_extras=(["bert-large-uncased:questionanswering"]="--batchsize 4" ["t5-large:lm"]="--batchsize 4")
declare -A server_extras=(["bert-large-uncased:questionanswering"]="--batchsize 4" ["t5-large:lm"]="--batchsize 4")

models="bert-base-uncased:questionanswering"
# models="$models bert-large-uncased:questionanswering distilbert-base-uncased-distilled-squad:questionanswering"
# models="$models google/bigbird-roberta-base:lm t5-small:lm t5-base:lm t5-large:lm distilbert-base-uncased:lm bert-base-uncased:lm albert-base-v2:lm facebook/bart-base:lm microsoft/deberta-v3-base:lm"

for model in $models; do
    for rt in "$pt" "$ort"; do
        extra=${singlestream_extras[$model]}
        python hf.py --model $model $single $opt $rt $extra $@
        extra=${offline_extras[$model]}
        python hf.py --model $model $offline $opt $rt $extra $@
    done
done

cat hf/summary.csv
