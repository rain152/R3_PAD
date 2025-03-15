#!/bin/bash

# 定义 classnames 列表
classnames=("01Gorilla" "02Unicorn" "03Mallard" "04Turtle" "05Whale" "06Bird" "07Owl" "08Sabertooth"
            "09Swan" "10Sheep" "11Pig" "12Zalika" "13Pheonix" "14Elephant" "15Parrot" "16Cat" "17Scorpion"
            "18Obesobeso" "19Bear" "20Puppy")
            
# 遍历列表并传递参数
for name in "${classnames[@]}"; do
    echo "Processing: $name"
    python train_lightnet.py -c $name
    # python main.py -c $name -skip_train -skip_loc
    # python main_draw.py -c $name -skip_train -skip_loc
    python main_ours.py -c $name -skip_train -skip_loc
    # python main_ours_draw.py -c $name -skip_train -skip_loc
done