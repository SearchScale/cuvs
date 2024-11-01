export CMAKE_PREFIX_PATH=`pwd`/../cpp/build
cd api-sys
cmake .
cmake --build .
cd ..
mvn install:install-file -DgroupId=ai.rapids.cuvs -DartifactId=api-sys -Dversion=0.1 -Dpackaging=so -Dfile=./api-sys/libcuvs_wrapper.so

cd api
mvn package
