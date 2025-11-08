MODEL="google/gemma3-3n-e4b"
NUMBER=100
PARALLEL=2

evalscope perf \
    --url "http://192.168.1.5:1234/v1/chat/completions" \
    --parallel ${PARALLEL} \
    --model ${MODEL} \
    --number ${NUMBER} \
    --api openai \
    --dataset openqa \
    --stream \
    --swanlab-api-key 'BIYVGq2rfWmD9sFMCehUG' \
    --name "AMD370-${MODEL}-number${NUMBER}-parallel${PARALLEL}"