ARCH=`arch`
CURRENT=$(cd $(dirname $0);pwd)
DIR_NAME=`echo "$1" | sed -e 's/.*\/\([^\/]*\)$/\1/'`
TARGET=$DIR_NAME

cd $CURRENT/$TARGET

DEPLOY="deploy_"$TARGET
SAMPLE=$TARGET"_sample"
SAMPLEMAIN=$TARGET"_Main"

#make directorys
mkdir -p $DEPLOY/data
mkdir -p $DEPLOY/include/kros/trt/TRT
mkdir -p $DEPLOY/include/kros/trt/$TARGET
mkdir -p $DEPLOY/lib/${ARCH}
mkdir -p $DEPLOY/bin
mkdir -p $DEPLOY/linux/$TARGET/benchmark
mkdir -p $DEPLOY/linux/utilities
mkdir -p $DEPLOY/module/$SAMPLE

#copy include headers
cp    ../../include/kros/trt.h $DEPLOY/include/kros/
cp -r ../../include/kros/trt/TRT/*.h $DEPLOY/include/kros/trt/TRT/
cp -r ../../include/kros/trt/$TARGET/*.h $DEPLOY/include/kros/trt/$TARGET/

#copy library
LIB_NAME=`echo libKROS_$TARGET.so | tr [A-Z] [a-z]`
cp    ../../bin/gcc_${ARCH}_Release/$LIB_NAME $DEPLOY/lib/${ARCH}

#copy shell&Makefile
cp    ../deploy/Makefile          $DEPLOY/linux/$TARGET/
sed -i -e "s/##MODEL##/${TARGET}/g" $DEPLOY/linux/$TARGET/Makefile

cp    ../deploy/Makefile_benchmark  $DEPLOY/linux/$TARGET/benchmark/Makefile
sed -i -e "s/##MODEL##/${TARGET}/g" $DEPLOY/linux/$TARGET/benchmark/Makefile

cp    ../deploy/install.sh        $DEPLOY/linux/$TARGET/
sed -i -e "s/##MODEL##/${TARGET}/g" $DEPLOY/linux/$TARGET/install.sh     
sed -i -e "s/##MODEL_LIB##/${LIB_NAME}/g" $DEPLOY/linux/$TARGET/install.sh     

cp    run.sh                      $DEPLOY/linux/$TARGET/
sed -i -e "s/${TARGET}/${TARGET}_sample/g" $DEPLOY/linux/$TARGET/run.sh     
sed -i -e "s/gcc_//g" $DEPLOY/linux/$TARGET/run.sh 
sed -i -e "s/_Release//g" $DEPLOY/linux/$TARGET/run.sh 

cp    run_benchmark.sh           $DEPLOY/linux/$TARGET/
sed -i -e "s/gcc_//g" $DEPLOY/linux/$TARGET/run_benchmark.sh 
sed -i -e "s/_Release//g" $DEPLOY/linux/$TARGET/run_benchmark.sh 

cp    ../deploy/uninstall.sh      $DEPLOY/linux/$TARGET/
sed -i -e "s/##MODEL##/${TARGET}/g" $DEPLOY/linux/$TARGET/uninstall.sh    
sed -i -e "s/##MODEL_LIB##/${LIB_NAME}/g" $DEPLOY/linux/$TARGET/uninstall.sh     

cp -r ../utilities/gpu_memory_checker.sh $DEPLOY/linux/utilities/

#copy sample source
cp    ../../module/$SAMPLEMAIN/$TARGET"_main.cpp" $DEPLOY/module/$SAMPLE/
cp    ../../module/$SAMPLEMAIN/$TARGET"_benchmark.cpp" $DEPLOY/module/$SAMPLE/
cp    ../../module/utilities.h $DEPLOY/module/

#archive
7z a ../../data/deploy_$TARGET.7z $DEPLOY
rm -rf $DEPLOY
