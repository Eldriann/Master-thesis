#!/bin/bash

SENTENCE_FILE="sentence.txt"
SUBJECT_FILE="guardian/stripped.txt"
TIME_FILE="time.txt"

OUTPUT_FILE="generated.txt"

NUM_PLACEHOLDER="__NUM__"
SUBJECT_PLACEHOLDER="__SUBJECT__"
TIME_PLACEHOLDER="__TIME__"

sentences_array=()
while IFS= read -r line || [[ "$line" ]]; do
    sentences_array+=("$line")
done < $SENTENCE_FILE

subject_array=()
while IFS= read -r line || [[ "$line" ]]; do
    subject_array+=("$line")
done < $SUBJECT_FILE

time_array=()
while IFS= read -r line || [[ "$line" ]]; do
    time_array+=("$line")
done < $TIME_FILE

RANDOM=$$$(date +%s)

rm $OUTPUT_FILE

for subject in "${subject_array[@]}"; do
    sentence_template=${sentences_array[$RANDOM % ${#sentences_array[@]}]}
    if [[ "$sentence_template" == *"$TIME_PLACEHOLDER"* ]]; then
        time_template=${time_array[$RANDOM % ${#time_array[@]}]}
        if [[ "$time_template" == *"$NUM_PLACEHOLDER"* ]]; then
            num=$(jot -w %i -r 1 2 8)
            sentence=$(echo ${sentence_template//$SUBJECT_PLACEHOLDER/"[$subject](topic)"})
            time=$(echo ${time_template//$NUM_PLACEHOLDER/$num})
            sentence=$(echo ${sentence//$TIME_PLACEHOLDER/"[$time](temporal)"})
            echo "- $sentence" >> $OUTPUT_FILE
        else
            sentence=$(echo ${sentence_template//$SUBJECT_PLACEHOLDER/"[$subject](topic)"})
            sentence=$(echo ${sentence//$TIME_PLACEHOLDER/"[$time_template](temporal)"})
            echo "- $sentence" >> $OUTPUT_FILE
        fi
    else
        sentence=$(echo ${sentence_template//$SUBJECT_PLACEHOLDER/"[$subject](topic)"})
        echo "- $sentence" >> $OUTPUT_FILE
    fi
done