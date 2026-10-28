# BF16 baseline corresponding to:
# /data/z00659619/run_logs/gpu_alignment_perf_20261020/benchmark_console_bf16_npu/output/v0-20261020-000829
# test_env: Ascend 950PR, 4 NPUs

MODEL=${MODEL:-/data/z00659619/.cache/modelscope/models/Qwen--Qwen3-14B-BF16}
DATASET=${DATASET:-/data/z00659619/gpu_processed_final_order-3000.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-output/Qwen3-14B-BF16-benchmark}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

NPROC_PER_NODE=4 \
megatron sft \
    --model "$MODEL" \
    --model_type qwen3 \
    --template qwen3 \
    --save_safetensors false \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --seed 42 \
    --data_seed 42 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --truncation_strategy delete \
    --strict false \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --padding_free true \
    --recompute_granularity selective \
    --train_iters 20 \
    --finetune true \
    --bf16 true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl native \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir "$OUTPUT_DIR" \
    --eval_steps 200 \
    --save_steps 0 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_precision_aware_optimizer true \
    --exp_avg_dtype bf16 \
    --exp_avg_sq_dtype bf16 \
    --attention_backend flash
