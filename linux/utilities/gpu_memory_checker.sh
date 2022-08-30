gpus=`nvidia-smi | grep 'MiB'| awk '{print $9}' | sed 's/MiB//g' | tr '\n' ',' | cut -d ',' -f 1 `
echo $gpus
