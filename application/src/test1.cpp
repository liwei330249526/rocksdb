#include <iostream>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

int main() {
  rocksdb::DB* db;
  rocksdb::Options options;
  options.create_if_missing = true;

  // 打开数据库
  rocksdb::Status status = rocksdb::DB::Open(options, "/tmp/testdb", &db);
  if (!status.ok()) {
    std::cerr << "无法打开数据库: " << status.ToString() << std::endl;
    return -1;
  }

  // 插入键值对
  status = db->Put(rocksdb::WriteOptions(), "key1", "value1");
  if (!status.ok()) {
    std::cerr << "插入失败: " << status.ToString() << std::endl;
    delete db;
    return -1;
  }

  // 读取键值对
  std::string value;
  status = db->Get(rocksdb::ReadOptions(), "key1", &value);
  if (status.ok()) {
    std::cout << "读取成功: key1 -> " << value << std::endl;
  } else {
    std::cerr << "读取失败: " << status.ToString() << std::endl;
  }

  // 关闭数据库
  delete db;
  return 0;
}