# -----------------------mono base-----------------------
# citeseer
python -u training.py --dataset citeseer --base mono --lr 0.0005 --weight_decay 0.001 --attn_lr 0.001 --attn_wd 1e-05 --hidden 256 --dropout 0.9 --dprate 0.6 --n_head 4 --d_ffn 128 --q 2.0 --multi 2.0 --K 6 --nlayer 1 

# cs
python -u training.py --dataset cs --base mono --lr 0.001 --weight_decay 1e-07 --attn_lr 0.005 --attn_wd 0.0001 --hidden 128 --dropout 0.0 --dprate 0.8 --n_head 8 --d_ffn 128 --q 1.0 --multi 1.0 --K 2 --nlayer 1 

# pubmed
python -u training.py --dataset pubmed --base mono --lr 0.005 --weight_decay 0.001 --attn_lr 0.0005 --attn_wd 1e-08 --hidden 256 --dropout 0.5 --dprate 0.8 --n_head 8 --d_ffn 32 --q 1.6 --multi 2.0 --K 2 --nlayer 2 

# physics
python -u training.py --dataset physics --base mono --lr 0.001 --weight_decay 1e-05 --attn_lr 0.0001 --attn_wd 0.001 --hidden 128 --dropout 0.9 --dprate 0.8 --n_head 2 --d_ffn 256 --q 1.2 --multi 0.5 --K 4 --nlayer 1 

# filtered chameleon
python -u training.py --dataset chameleon_filtered --base mono --lr 0.01 --weight_decay 1e-06 --attn_lr 5e-05 --attn_wd 1e-05 --hidden 256 --dropout 0.2 --dprate 0.4 --n_head 8 --d_ffn 32 --q 2.0 --multi 2.0 --K 6 --nlayer 4 

# filtered squirrel 
python -u training.py --dataset squirrel_filtered --base mono --lr 0.0001 --weight_decay 0.0 --attn_lr 0.001 --attn_wd 1e-07 --hidden 256 --dropout 0.3 --dprate 0.8 --n_head 4 --d_ffn 128 --q 1.4 --multi 1.0 --K 12 --nlayer 2 

# minesweeper
python -u training.py --dataset minesweeper --base mono --lr 0.01 --weight_decay 1e-05 --attn_lr 0.0005 --attn_wd 0.0 --hidden 16 --dropout 0.2 --dprate 0.3 --n_head 8 --d_ffn 32 --q 1.6 --multi 2.0 --K 10 --nlayer 4 

# tolokers
python -u training.py --dataset tolokers --base mono --lr 0.0005 --weight_decay 1e-08 --attn_lr 0.0001 --attn_wd 0.0 --hidden 64 --dropout 0.2 --dprate 0.0 --n_head 16 --d_ffn 128 --q 1.0 --multi 1.0 --K 10 --nlayer 1 

# roman-empire
python -u training.py --dataset roman-empire --base mono --lr 0.0001 --weight_decay 0.001 --attn_lr 0.001 --attn_wd 0.0001 --hidden 256 --dropout 0.5 --dprate 0.1 --n_head 16 --d_ffn 64 --q 1.0 --multi 2.0 --K 14 --nlayer 3 

# questions
python -u training.py --dataset questions --base mono --lr 0.0005 --weight_decay 0.001 --attn_lr 0.0001 --attn_wd 0.0 --hidden 128 --dropout 0.2 --dprate 0.3 --n_head 4 --d_ffn 256 --q 1.0 --multi 1.0 --K 12 --nlayer 1 



# -----------------------cheb base-----------------------
# citeseer
python -u training.py --dataset citeseer --base cheb --lr 0.0005 --weight_decay 1e-05 --attn_lr 0.001 --attn_wd 1e-05 --hidden 256 --dropout 0.7000000000000001 --dprate 0.4 --n_head 1 --d_ffn 64 --q 1.6 --multi 0.5 --K 2 --nlayer 2 

# cs
python -u training.py --dataset cs --base cheb --lr 0.0005 --weight_decay 0.001 --attn_lr 0.01 --attn_wd 0.001 --hidden 64 --dropout 0.8 --dprate 0.0 --n_head 8 --d_ffn 32 --q 1.0 --multi 0.5 --K 4 --nlayer 1 

# pubmed
python -u training.py --dataset pubmed --base cheb --lr 0.0005 --weight_decay 0.001 --attn_lr 0.0005 --attn_wd 0.0001 --hidden 128 --dropout 0.30000000000000004 --dprate 0.0 --n_head 16 --d_ffn 64 --q 2.0 --multi 2.0 --K 6 --nlayer 1 

# physics
python -u training.py --dataset physics --base cheb --lr 0.001 --weight_decay 1e-05 --attn_lr 0.005 --attn_wd 0.001 --hidden 128 --dropout 0.9 --dprate 0.1 --n_head 8 --d_ffn 64 --q 1.6 --multi 0.5 --K 2 --nlayer 2 

# filterd chameleon
python -u training.py --dataset chameleon_filtered --base cheb --lr 0.01 --weight_decay 1e-07 --attn_lr 0.005 --attn_wd 1e-06 --early_stopping 30 --hidden 128 --dropout 0.6 --dprate 0.9 --n_head 16 --d_ffn 256 --q 1.4 --multi 0.5 --K 2 --nlayer 1 

# filtered squirrel 
python -u training.py --dataset squirrel_filtered --base cheb --lr 0.0005 --weight_decay 1e-07 --attn_lr 0.0005 --attn_wd 0.001 --hidden 256 --dropout 0.1 --dprate 0.8 --n_head 8 --d_ffn 256 --q 1.4 --multi 0.5 --K 12 --nlayer 1 

# minesweeper
python -u training.py --dataset minesweeper --base cheb --lr 0.0005 --weight_decay 1e-07 --attn_lr 0.0001 --attn_wd 0.0001 --hidden 128 --dropout 0.9 --dprate 0.3 --n_head 16 --d_ffn 128 --q 1.6 --multi 2.0 --K 8 --nlayer 2 

# tolokers
python -u training.py --dataset tolokers --base cheb --lr 0.001 --weight_decay 0.0001 --attn_lr 0.0001 --attn_wd 0.0001 --hidden 256 --dropout 0.0 --dprate 0.9 --n_head 16 --d_ffn 32 --q 1.6 --multi 2.0 --K 8 --nlayer 1 

# roman-empire
python -u training.py --dataset roman-empire --base cheb --lr 0.005 --weight_decay 1e-05 --attn_lr 0.0005 --attn_wd 1e-08 --hidden 256 --dropout 0.5 --dprate 0.3 --n_head 16 --d_ffn 64 --q 1.4 --multi 2.0 --K 2 --nlayer 3 

# questions
python -u training.py --dataset questions --base cheb --lr 0.0001 --weight_decay 0.0 --attn_lr 0.0005 --attn_wd 1e-05 --hidden 256 --dropout 0.4 --dprate 0.7 --n_head 16 --d_ffn 128 --q 1.6 --multi 2.0 --K 10 --nlayer 2 


