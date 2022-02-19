# Install script for directory: /home/hena/caffe-ocr/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/hena/caffe-ocr/buildcmake/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification"
         RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/hena/caffe-ocr/buildcmake/examples/cpp_classification/classification")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification"
         OLD_RPATH "/usr/local/cuda-8.0/lib64:/home/hena/caffe-ocr/buildcmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib::::::::"
         NEW_RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data"
         RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/hena/caffe-ocr/buildcmake/examples/cifar10/convert_cifar_data")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data"
         OLD_RPATH "/usr/local/cuda-8.0/lib64:/home/hena/caffe-ocr/buildcmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib::::::::"
         NEW_RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data"
         RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/hena/caffe-ocr/buildcmake/examples/mnist/convert_mnist_data")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data"
         OLD_RPATH "/usr/local/cuda-8.0/lib64:/home/hena/caffe-ocr/buildcmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib::::::::"
         NEW_RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data"
         RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/hena/caffe-ocr/buildcmake/examples/siamese/convert_mnist_siamese_data")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data"
         OLD_RPATH "/usr/local/cuda-8.0/lib64:/home/hena/caffe-ocr/buildcmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib::::::::"
         NEW_RPATH "/home/hena/caffe-ocr/buildcmake/install/lib:/usr/local/cuda-8.0/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/home/hena/tool/protobuf-3.1.0/build/install/lib:/home/hena/tool/opencv-3.2.0/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data")
    endif()
  endif()
endif()

