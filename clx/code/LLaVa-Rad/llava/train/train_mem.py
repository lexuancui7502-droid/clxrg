# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# 在导入 Transformers / LLaMA 前先对 LLaMA 的注意力模块做 monkey-patch，替换成 FlashAttention 的实现，以降低显存/内存占用并加速注意力计算
# replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
