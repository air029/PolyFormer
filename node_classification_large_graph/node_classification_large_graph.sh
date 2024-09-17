# gamer
python training_batch.py --lr 5e-05 --weight_decay 0.0005 --attn_lr 0.0002 --attn_wd 5e-05 --early_stopping 250 --hidden 128 --dropout 0.0 --n_head 4 --d_ffn 512 --K 8 --nlayer 2 --base mono --dprate 0.6 --batch_size 10000 --q 1.0 --multi 0.5 --dataset twitch-gamer --device_idx 0 --net PolyFormer --runs 1 --test 1 --metric acc 

# pokec
python training_batch.py --lr 0.0002 --weight_decay 5e-05 --attn_lr 0.0002 --attn_wd 0.0005 --early_stopping 250 --hidden 512 --dropout 0.25 --n_head 1 --d_ffn 1024 --K 4 --nlayer 1 --base mono --dprate 0.0 --batch_size 20000 --q 2.0 --multi 1.0 --dataset pokec --device_idx 0 --net PolyFormer --runs 1 --test 1 --metric acc 

# arxiv
python training_batch.py --lr 0.0002 --weight_decay 0.0 --attn_lr 0.01 --attn_wd 0.001 --early_stopping 250 --hidden 128 --dropout 0.4 --n_head 8 --d_ffn 512 --K 10 --nlayer 2 --base mono --dprate 0.6 --batch_size 10000 --q 2.0 --multi 0.5 --dataset ogbn-arxiv --device_idx 0 --net PolyFormer --runs 1 --test 1 --metric acc 

# papers100M
python training_batch.py --lr 0.0002 --weight_decay 0.0 --attn_lr 5e-05 --attn_wd 0.0 --early_stopping 250 --hidden 1024 --dropout 0.5 --n_head 1 --d_ffn 512 --K 4 --nlayer 1 --base mono --dprate 0.8 --batch_size 50000  --q 1.0 --multi 1.0 --dataset ogbn-papers100M --device_idx 0 --net PolyFormer --runs 1  --test 1 --metric acc 
