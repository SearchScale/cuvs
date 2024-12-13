export CMAKE_PREFIX_PATH=`pwd`/../cpp/build
VERSION="25.02"
cd internal && cmake . && cmake --build . \
  && cd .. \
  && mvn install:install-file -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=./internal/libcuvs_java.so \
  && mvn install:install-file -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=./internal/libcuvs_java_brute_force.so \
  && cd cuvs-java \
  && mvn -DskipTests=true package \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION-jar-with-dependencies.jar -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar
