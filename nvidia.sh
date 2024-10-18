#!/bin/bash

# Define paths
HOST_MODEL_DIR="/data0/MODELS/nvidia/Llama3-70B-SteerLM-RM"
CONTAINER_MODEL_DIR="/models/nvidia"



# Docker image
DOCKER_IMAGE="nvcr.io/nvidia/nemo:24.01.framework"

# Inference server port
INFERENCE_PORT=1424

# Run the Docker container and start the server
docker run --gpus '"device=4,5"' \
    --runtime=nvidia \
    --rm \
    -it \
    --name lyt_nemo \
    --network=host \
    -v $HOST_MODEL_DIR:$CONTAINER_MODEL_DIR \
    -v data:/workspace/data \
    -v $HF_HOME:/workspace/hf \
    -e NCCL_DEBUG=INFO \
    -e NCCL_DEBUG_SUBSYS=ALL \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_P2P_LEVEL=1 \
    -e http_proxy="http://localhost:8001" \
    -e https_proxy="http://localhost:8001" \
    -e no_proxy="localhost,127.0.0.1" \
    --shm-size=200g \
    $DOCKER_IMAGE \
    /bin/bash

# export CONTAINER_MODEL_DIR="/models/nvidia" && export HF_HOME=/workspace/hf && export INFERENCE_PORT=1424 && python /opt/NeMo-Aligner/examples/nlp/gpt/serve_reward_model.py \
#         rm_model_file=$CONTAINER_MODEL_DIR \
#         trainer.num_nodes=1 \
#         trainer.devices=2 \
#         ++model.tensor_model_parallel_size=1 \
#         ++model.pipeline_model_parallel_size=2 \
#         inference.micro_batch_size=2 \
#         inference.port=$INFERENCE_PORT


# python /opt/NeMo-Aligner/examples/nlp/data/steerlm/attribute_annotate.py \
#       --input-file=data/our-bench/step5_v5_chat_factual_construct.jsonl \
#       --output-file=data/our-bench/step5_v5_chat_factual_construct_labeled.jsonl \
#       --port=1424


# docker run --gpus '"device=0"' \
#     --rm \
#     --name nemo_inference_server \
#     -p $INFERENCE_PORT:$INFERENCE_PORT \
#     -v $HOST_MODEL_DIR:$CONTAINER_MODEL_DIR \
#     $DOCKER_IMAGE \
#     python /opt/NeMo-Aligner/examples/nlp/gpt/serve_reward_model.py \
#         rm_model_file=$CONTAINER_MODEL_DIR/Llama2-13B-SteerLM-RM.nemo \
#         trainer.num_nodes=1 \
#         trainer.devices=1 \
#         ++model.tensor_model_parallel_size=1 \
#         ++model.pipeline_model_parallel_size=1 \
#         inference.micro_batch_size=2 \
#         inference.port=$INFERENCE_PORT

# python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst

# python /opt/NeMo-Aligner/examples/nlp/data/steerlm/attribute_annotate.py \
#       --input-file=data/oasst/train.jsonl \
#       --output-file=data/oasst/train_labeled.jsonl \
#       --port=1424