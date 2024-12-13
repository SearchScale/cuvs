export CMAKE_PREFIX_PATH=`pwd`/../cpp/build

VERSION="25.02"
GROUP_ID="com.nvidia.cuvs"
SO_FILE_PATH="./internal"

cd internal && cmake . && cmake --build . \
  && cd .. \
  && mvn install:install-file -DgroupId=$GROUP_ID -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=$SO_FILE_PATH/libcuvs_java_cagra.so \
  && mvn install:install-file -DgroupId=$GROUP_ID -DartifactId=cuvs-java-internal -Dversion=$VERSION -Dpackaging=so -Dfile=$SO_FILE_PATH/libcuvs_java_brute_force.so \
  && cd cuvs-java \
  && mvn -DskipTests=false package \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION-jar-with-dependencies.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar
