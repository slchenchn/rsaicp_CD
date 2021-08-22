source env.sh
# python tools/train.py configs/siamunet/unet_512x512_40k_s2looking.py 
# python tools/train.py configs/siamunet/unet_512x512_20k_s2looking.py \
    # --options optimizer.lr=1e-5

export CUDA_VISIBLE_DEVICES=1,2,3


