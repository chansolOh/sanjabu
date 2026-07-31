#!/bin/bash

start_flag=false

env="Logistic_site"
time_th=120
if [ "$env" = "Home" ];then
    time_th=400
else
    time_th=300
fi

python_file="./sanjabu_scene_generator.py"
python_path="/home/cubox/.local/share/virtualenvs/isaac_code-SQV0LN_U/bin/python"
data_dir="/nas/Dataset/Dataset_2025/test_data/$env/conf"

max=0

for file in "$data_dir"/*.json; do
    name=$(basename "$file" .json)

    if [[ "$name" =~ ^[0-9]+$ ]]; then
        num=$((10#$name))  # 앞에 0 있어도 안전한 10진수 처리
        if (( num > max )); then
            max=$num
        fi
    fi
done

# max=$((max + 1))


replay=false

while true; do
    time_count=0
    $python_path $python_file --scene_num $max --env $env & #> /dev/null 2>&1 &
    pid=$!

    while true; do
        if [ -f "$data_dir/$(printf "%04d" "$max").json" ]; then
            time_count=0
            echo -e "\033[34m파일생성 확인 완료\033[0m"



            for file in "$data_dir"/*.json; do
                name=$(basename "$file" .json)

                if [[ "$name" =~ ^[0-9]+$ ]]; then
                    num=$((10#$name))  # 앞에 0 있어도 안전한 10진수 처리
                    if (( num > max )); then
                        max=$num
                    fi
                fi
            done
            max=$((max + 1))
            echo -e "\033[32m max값 = $max \033[0m"




        else
            time_count=$((time_count + 1))
            echo -e "\033[32m time_count = $time_count \033[0m"
            sleep 1 
        fi

        if [ "$time_count" -ge "$time_th" ]; then
            replay=true
            echo "5분 동안 파일이 생성되지 않았습니다. 다시 시도합니다."
            break
        fi
    done

    if [ "$replay" = true ]; then
        kill -9 $(ps -ef | grep "$python_file" | grep -v "grep" | awk '{print $2}')
        replay=false
        sleep 20
        continue
    fi

    if [ "$max" -eq 1000 ]; then
        echo "1000개 생성 완료"
        break
    fi


done
