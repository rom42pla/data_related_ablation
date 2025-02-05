python generate_configs.py configs
for filename in `ls configs`; do
    if [[ $filename =~ $1 ]];
    then
        python main.py configs/$filename
        bash 10
    else
        echo "skipping '$filename' as it does not match '$1'"
    fi
done