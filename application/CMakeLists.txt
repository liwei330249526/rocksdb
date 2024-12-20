set(MYDB_NAME myrocksdb)  # 设置一个名字

include_directories(include) # 包含头文件
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

message("编译我的 rocksdb")
message(${PROJECT_SOURCE_DIR}/include)
message(${PROJECT_SOURCE_DIR}/include)
file(GLOB ROCKSDB_WRAPPER_SRC src/*.cpp)  # 包含源文件
add_executable(${MYDB_NAME} ${ROCKSDB_WRAPPER_SRC}) # 可执行文件

# 查找 RocksDB 库
#find_package(${ROCKSDB_LIB} REQUIRED)

# 链接 RocksDB 库
target_link_libraries(${MYDB_NAME} PRIVATE ${ROCKSDB_STATIC_LIB}) # 链接静态库

#target_link_libraries(${ROCKSDB_LIB}) # 连接库
