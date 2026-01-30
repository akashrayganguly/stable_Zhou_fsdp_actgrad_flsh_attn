This stable version implements the following onto the Informer2020 (https://github.com/zhouhaoyi/Informer2020):
1. Fully Sharded FSDP using Pytorch
2. Flash Attention replacing full attention class
3. Gradient Accumulation
4. Activation Checkpointing
5. Mixed Precision (in exp_informer) BF16, FP16 depending on the GPU architecture available
