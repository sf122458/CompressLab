cd BPG
make -j16
cd ../HM
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
cd ../../VTM
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
cd ../..