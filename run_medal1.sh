MODEL_LIST=(gpt-4o-2024-08-06 gpt-4o-2024-05-13 gpt-4o-mini-2024-07-18 gpt-4-turbo-2024-04-09 gpt-3.5-turbo-0125 gemini-1.5-pro-001 gemini-1.5-flash-001 claude-3-5-sonnet-20240620 claude-3-sonnet-20240229 claude-3-haiku-20240307 mistral-large-2407 open_mistral-nemo-2407)
# MODEL_LIST=(meta-llama/Meta-Llama-3.1-8B-Instruct)
FILE_NAME=./data/question_medal1.json
#OUTPUT_FILE=./result/answer_medal1_$MODEL_NAME.json

# vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype auto --max-model-len 2048 --max-num-seqs 1


for MODEL_NAME in ${MODEL_LIST[@]}; do
    # If / in MODEL_NAME, replace it with _
    OUTPUT_FILE=./result/answer_medal1_${MODEL_NAME//\//_}.json
    python answer_langchain_medal1.py --model_name $MODEL_NAME --data_file $FILE_NAME --output_file $OUTPUT_FILE
done
