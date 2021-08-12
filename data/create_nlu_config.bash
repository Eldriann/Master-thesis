#!/bin/bash

SENTENCE_FILE="sentence.txt"
SUBJECT_FILE="subject.txt"
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

rm $OUTPUT_FILE
for sentence_template in "${sentences_array[@]}"; do
    if [[ "$sentence_template" == *"$SUBJECT_PLACEHOLDER"* ]]; then
        for subject in "${subject_array[@]}"; do
            if [[ "$sentence_template" == *"$TIME_PLACEHOLDER"* ]]; then
                for time_template in "${time_array[@]}"; do
                    if [[ "$time_template" == *"$NUM_PLACEHOLDER"* ]]; then
                        for num in $(seq 2 7); do
                            sentence=$(echo ${sentence_template//$SUBJECT_PLACEHOLDER/"[$subject](topic)"})
                            time=$(echo ${time_template//$NUM_PLACEHOLDER/$num})
                            sentence=$(echo ${sentence//$TIME_PLACEHOLDER/"[$time](temporal)"})
                            echo "- $sentence" >> $OUTPUT_FILE
                        done
                    else
                        sentence=$(echo ${sentence_template//$SUBJECT_PLACEHOLDER/"[$subject](topic)"})
                        sentence=$(echo ${sentence//$TIME_PLACEHOLDER/"[$time_template](temporal)"})
                        echo "- $sentence" >> $OUTPUT_FILE
                    fi
                done
            else
                sentence=$(echo ${sentence_template//$SUBJECT_PLACEHOLDER/"[$subject](topic)"})
                echo "- $sentence" >> $OUTPUT_FILE
            fi
        done
    fi
done