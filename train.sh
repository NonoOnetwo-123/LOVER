# training model without sample filtering per epoch

export WANDB_PROJECT=CL-23-long-short-754-mid-768
export EXP_NAME=23-long-short-754-mid-768
export PYTHONPATH=".:$PYTHONPATH"
export DEBUG_MODE="true"
export LOG_PATH="./logs/$EXP_NAME/$EXP_NAME.txt"

OUTDIR=/data/chenkaiyan/23-long-short-754-mid-768
BASE_MODEL_NAME_OR_PATH="/data/chenkaiyan/129-short-mid-768/checkpoint-754"

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12399" \
    main_no_num.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --train_data_path /home/chenkaiyan/time-R1/time-r1/dataset/timer1/annotations/output_long_with_choices.json \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 128 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 2 \
    --run_name $EXP_NAME \
    --report_to tensorboard \
    --reward_funcs choice iop format \
    --temperature 1.0 \
    --prompt_type gqa_v1 \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 487 \
    --save_only_model true
