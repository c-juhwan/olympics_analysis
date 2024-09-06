MODEL_LIST=(gpt-4o-2024-08-06 gpt-4o-2024-05-13 gpt-4o-mini-2024-07-18 gpt-4-turbo-2024-04-09 gpt-3.5-turbo-0125 gemini-1.5-pro-001 gemini-1.5-flash-001 claude-3-5-sonnet-20240620 claude-3-sonnet-20240229 claude-3-haiku-20240307 mistral-large-2407 open_mistral-nemo-2407)
FILE_NAME=./data/question_team1.json
OUTPUT_FILE=./result/answer_team1_$MODEL_NAME.json

for MODEL_NAME in ${MODEL_LIST[@]}; do
    OUTPUT_FILE=./result/answer_team1_$MODEL_NAME.json
    python answer_langchain_team1.py --model_name $MODEL_NAME --data_file $FILE_NAME --output_file $OUTPUT_FILE
done
