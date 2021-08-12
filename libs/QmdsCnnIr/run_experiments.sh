# Required environment variables:
# DATASET: choose the dataset (CNNDM / WIKI)
# MODEL_TYPE: choose the type of model (hier, he, order, query, heq, heo, hero)

BATCH_SIZE=8000
SEED=666
TRAIN_STEPS=500000
SAVE_CHECKPOINT_STEPS=5000
REPORT_EVERY=100
VISIBLE_GPUS="0,1,2,3"
GPU_RANKS="0,1,2,3"
WORLD_SIZE=4
ACCUM_COUNT=2
DROPOUT=0.1
LABEL_SMOOTHING=0.1
INTER_LAYERS="6,7"
INTER_HEADS=8
LR=1
MAX_SAMPLES=500

case $MODEL_TYPE in
    query|heq|hero)
        QUERY=True
        ;;
    hier|he|order|heo)
        QUERY=False
        ;;
    *)
        echo "Invalid option: ${MODEL_TYPE}"
        ;;
esac

case $DATASET in
    CNNDM)
        TRUNC_TGT_NTOKEN=100
        TRUNC_SRC_NTOKEN=200
        TRUNC_SRC_NBLOCK=8
        if [ $QUERY == "False" ]; then
            DATA_FOLDER_NAME=pytorch_qmdscnn
        else
            DATA_FOLDER_NAME=pytorch_qmdscnn_query
        fi
        if [ -z ${DATA_PATH+x} ]; then 
            DATA_PATH="data/qmdscnn/${DATA_FOLDER_NAME}/CNNDM"
        fi
        if [ -z ${VOCAB_PATH+x} ]; then 
            VOCAB_PATH="data/qmdscnn/${DATA_FOLDER_NAME}/spm.model"
        fi
        ;;
    WIKI)
        TRUNC_TGT_NTOKEN=400
        TRUNC_SRC_NTOKEN=100
        TRUNC_SRC_NBLOCK=24
        if [ $QUERY == "False" ]; then
            DATA_FOLDER_NAME=ranked_wiki_b40
        else
            DATA_FOLDER_NAME=ranked_wiki_b40_query
        fi
        if [ -z ${DATA_PATH+x} ]; then 
            DATA_PATH="data/wikisum/${DATA_FOLDER_NAME}/WIKI"
        fi
        if [ -z ${VOCAB_PATH+x} ]; then 
            VOCAB_PATH="data/wikisum/${DATA_FOLDER_NAME}/spm9998_3.model"
        fi
        ;;
    *)
        echo "Invalid option: ${DATASET}"

esac

# If model path not set
if [ -z ${MODEL_PATH+x} ]; then
    MODEL_PATH="results/model-${DATASET}-${MODEL_TYPE}"
fi


# python src/train_abstractive.py -mode train -batch_size 8000 -seed 666 -train_steps 500000 -save_checkpoint_steps 5000 -report_every 100 -trunc_tgt_ntoken 100 -trunc_src_ntoken 200 -trunc_src_nblock 8 -visible_gpus "0" -gpu_ranks "0" -world_size 1 -accum_count 2 -lr 1 -dec_dropout 0.1 -enc_dropout 0.1 -label_smoothing 0.1 -inter_layers "6,7" -inter_heads 8 -hier -dataset "CNNDM" -model_type "he" -query False -max_samples 500 -data_path "data/qmdscnn/pytorch_QMDS_adv/CNNDM" -vocab_path "data/qmdscnn/pytorch_QMDS_adv/spm.model" -model_path "results/model-CNNDM-he" -result_path "results/model-CNNDM-he/outputs" 

