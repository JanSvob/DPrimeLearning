# Name of the model that is being tested
modelName="/media/nas/palmprints_cnn/arch_17_8_siam_c"
# Path to the CNN architecture that is being used in the model
archName="Models/Architectures/arch_17"
# Output directory to save the test results into
outputDir="arch_17_8_c"
# Which GPU should be used
gpu="gpu2"
# Name of the dataset to evaluate
dataset="Casia"
# Normalize the input samples before passing to the CNN
normalize="--no-norm_input"
# Output feature vector size
outputSize="32"
# Network parameters
dimWidth="136"  # Input image width
dimHeight="136" # Input image height
dimDepth="1"    # Input image depth (channels)
# How many pixels to crop for the augmentation
cropSize="8"
# Training epoch from which to take the model parameters
epoch="400"
# Dataset train/test splitting parameters
splitMod="6"        # label % splitMod
splitTrain="1,4,5"      # label % splitMod in splitTrain (can contain more values -> 0,1,2)
splitTest="0,2,3"       # label % splitMod in splitTest (can contain more values -> 3,4,5)


# Run the Python test script with the preset parameters
eval "THEANO_FLAGS='device=$gpu,floatX=float32,optimizer_include=cudnn' python test.py --output_dir=$outputDir --arch=$archName --model=$modelName --dataset=$dataset --epoch=$epoch --dim_depth=$dimDepth --dim_width=$dimWidth --dim_height=$dimHeight --output_dim=$outputSize --crop_size=$cropSize $normalize --split_mod=$splitMod --split_train=$splitTrain --split_test=$splitTest"

