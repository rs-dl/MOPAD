configs="mopad grid cascade libra ga faster"
saveDir="results4A"
segImgDir="/data/zjp/SOPPM/SegImg/segImg4A"
for config in $configs
do
echo $config
# nohup bash tools/dist_train.sh configs/oilPalmUav/$config.py 2 >log/$config-0104.log 2>&1 &
mkdir /data/zjp/SOPPM/$saveDir/$config
python demoFull.py ../configs/oilPalmUav/$config.py ../work_dirs/$config/epoch_24.pth /data/zjp/SOPPM/$saveDir/$config/$config-det.txt $segImgDir
done
