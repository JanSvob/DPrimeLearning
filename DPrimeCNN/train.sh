# Name of the model that is being tested
modelName="arch_17_8_siam_c"
# Path to the CNN architecture that is being used in the model
archName="Models/Architectures/arch_17"
# Which GPU should be used
gpu="gpu2"
# Name of the dataset to evaluate
dataset="Casia"
# Loss function
loss="siamese"
# Normalize the input samples before passing to the CNN
normalize="--no-norm_input"
# Output feature vector size
outputSize="32"
# Training epoch parameters
batchSize="128" # How many samples (triplets) to have in each batch
numBatches="30" # How many batches to evaluate in each epohch
# Network parameters
dimWidth="136"  # Input image width
dimHeight="136" # Input image height
dimDepth="1"    # Input image depth (channels)
# How many pixels to crop for the augmentation
cropSize="8"
# Optimization parameters
margin="100"    # Margin for the negative distribution standard deviation
alpha="0.5"     # Weight between positive and negative scores (NOT USED NOW)
mu="1e-4"       # Regularization weight
# Dataset train/test splitting parameters
splitMod="6"        # label % splitMod
splitTrain="1,4,5"      # label % splitMod in splitTrain (can contain more values -> 0,1,2)
splitTest="0,2,3"       # label % splitMod in splitTest (can contain more values -> 3,4,5)


# Run the Python train script with the preset parameters
# Retraining after 100 epochs with decreasing learning rates
# In total the training now has 400 epochs (4 x 100 epochs)
eval "THEANO_FLAGS='device=$gpu,floatX=float32,optimizer_include=cudnn' python train.py --oper=train --arch=$archName --model=$modelName --dataset=$dataset --loss=$loss --num_epochs=100 --batch_size=$batchSize --num_batches=$numBatches --dim_depth=$dimDepth --dim_width=$dimWidth --dim_height=$dimHeight --output_dim=$outputSize --crop_size=$cropSize --dist_margin=$margin --mu=$mu --alpha=$alpha --lr=1e-2 $normalize --split_mod=$splitMod --split_train=$splitTrain --split_test=$splitTest"
eval "THEANO_FLAGS='device=$gpu,floatX=float32,optimizer_include=cudnn' python train.py --oper=retrain --arch=$archName --model=$modelName --dataset=$dataset --loss=$loss --num_epochs=100 --batch_size=$batchSize --num_batches=$numBatches --dim_depth=$dimDepth --dim_width=$dimWidth --dim_height=$dimHeight --output_dim=$outputSize --crop_size=$cropSize --dist_margin=$margin --mu=$mu --alpha=$alpha --lr=5e-3 --start_epoch=100 $normalize --split_mod=$splitMod --split_train=$splitTrain --split_test=$splitTest"
eval "THEANO_FLAGS='device=$gpu,floatX=float32,optimizer_include=cudnn' python train.py --oper=retrain --arch=$archName --model=$modelName --dataset=$dataset --loss=$loss --num_epochs=100 --batch_size=$batchSize --num_batches=$numBatches --dim_depth=$dimDepth --dim_width=$dimWidth --dim_height=$dimHeight --output_dim=$outputSize --crop_size=$cropSize --dist_margin=$margin --mu=$mu --alpha=$alpha --lr=1e-3 --start_epoch=200 $normalize --split_mod=$splitMod --split_train=$splitTrain --split_test=$splitTest"
eval "THEANO_FLAGS='device=$gpu,floatX=float32,optimizer_include=cudnn' python train.py --oper=retrain --arch=$archName --model=$modelName --dataset=$dataset --loss=$loss --num_epochs=100 --batch_size=$batchSize --num_batches=$numBatches --dim_depth=$dimDepth --dim_width=$dimWidth --dim_height=$dimHeight --output_dim=$outputSize --crop_size=$cropSize --dist_margin=$margin --mu=$mu --alpha=$alpha --lr=5e-4 --start_epoch=300 $normalize --split_mod=$splitMod --split_train=$splitTrain --split_test=$splitTest"

