#!/bin/bash

cd data
for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
mv ${ii}.mtx ${ii}.mt0
./conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

cd ..

