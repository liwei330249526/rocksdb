//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.  Use of
// this source code is governed by a BSD-style license that can be found
// in the LICENSE file. See the AUTHORS file for names of contributors.
//
// InlineSkipList is derived from SkipList (skiplist.h), but it optimizes
// the memory layout by requiring that the key storage be allocated through
// the skip list instance.  For the common case of SkipList<const char*,
// Cmp> this saves 1 pointer per skip list node and gives better cache
// locality, at the expense of wasted padding from using AllocateAligned
// instead of Allocate for the keys.  The unused padding will be from
// 0 to sizeof(void*)-1 bytes, and the space savings are sizeof(void*)
// bytes, so despite the padding the space used is always less than
// SkipList<const char*, ..>.
//
// Thread safety -------------
//
// Writes via Insert require external synchronization, most likely a mutex.
// InsertConcurrently can be safely called concurrently with reads and
// with other concurrent inserts.  Reads require a guarantee that the
// InlineSkipList will not be destroyed while the read is in progress.
// Apart from that, reads progress without any internal locking or
// synchronization.
//
// Invariants:
//
// (1) Allocated nodes are never deleted until the InlineSkipList is
// destroyed.  This is trivially guaranteed by the code since we never
// delete any skip list nodes.
//
// (2) The contents of a Node except for the next/prev pointers are
// immutable after the Node has been linked into the InlineSkipList.
// Only Insert() modifies the list, and it is careful to initialize a
// node and use release-stores to publish the nodes in one or more lists.
//
// ... prev vs. next pointer ordering ...
//

#pragma once
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <atomic>
#include <type_traits>
#include "memory/allocator.h"
#include "port/likely.h"
#include "port/port.h"
#include "rocksdb/slice.h"
#include "util/coding.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

template <class Comparator>
class InlineSkipList {
 private:
  struct Node;
  struct Splice;

 public:
  using DecodedKey = \
    typename std::remove_reference<Comparator>::type::DecodedType;

  static const uint16_t kMaxPossibleHeight = 32;

  // Create a new InlineSkipList object that will use "cmp" for comparing
  // keys, and will allocate memory using "*allocator".  Objects allocated
  // in the allocator must remain allocated for the lifetime of the
  // skiplist object.
  explicit InlineSkipList(Comparator cmp, Allocator* allocator,
                          int32_t max_height = 12,
                          int32_t branching_factor = 4);
  // No copying allowed
  InlineSkipList(const InlineSkipList&) = delete;
  InlineSkipList& operator=(const InlineSkipList&) = delete;

  // Allocates a key and a skip-list node, returning a pointer to the key
  // portion of the node.  This method is thread-safe if the allocator
  // is thread-safe. 分配一个键和一个跳链表节点，返回指向节点键部分的指针。如果分配器是线程安全的，那么此方法就是线程安全的。
  char* AllocateKey(size_t key_size);

  // Allocate a splice using allocator.
  Splice* AllocateSplice();

  // Allocate a splice on heap.
  Splice* AllocateSpliceOnHeap();

  // Inserts a key allocated by AllocateKey, after the actual key value
  // has been filled in.
  //
  // REQUIRES: nothing that compares equal to key is currently in the list.
  // REQUIRES: no concurrent calls to any of inserts.
  bool Insert(const char* key);

  // Inserts a key allocated by AllocateKey with a hint of last insert
  // position in the skip-list. If hint points to nullptr, a new hint will be
  // populated, which can be used in subsequent calls.
  //
  // It can be used to optimize the workload where there are multiple groups
  // of keys, and each key is likely to insert to a location close to the last
  // inserted key in the same group. One example is sequential inserts.
  //
  // REQUIRES: nothing that compares equal to key is currently in the list.
  // REQUIRES: no concurrent calls to any of inserts.
  bool InsertWithHint(const char* key, void** hint);

  // Like InsertConcurrently, but with a hint
  //
  // REQUIRES: nothing that compares equal to key is currently in the list.
  // REQUIRES: no concurrent calls that use same hint
  bool InsertWithHintConcurrently(const char* key, void** hint);

  // Like Insert, but external synchronization is not required.
  bool InsertConcurrently(const char* key);

  // Inserts a node into the skip list.  key must have been allocated by
  // AllocateKey and then filled in by the caller.  If UseCAS is true,
  // then external synchronization is not required, otherwise this method
  // may not be called concurrently with any other insertions.
  //
  // Regardless of whether UseCAS is true, the splice must be owned
  // exclusively by the current thread.  If allow_partial_splice_fix is
  // true, then the cost of insertion is amortized O(log D), where D is
  // the distance from the splice to the inserted key (measured as the
  // number of intervening nodes).  Note that this bound is very good for
  // sequential insertions!  If allow_partial_splice_fix is false then
  // the existing splice will be ignored unless the current key is being
  // inserted immediately after the splice.  allow_partial_splice_fix ==
  // false has worse running time for the non-sequential case O(log N),
  // but a better constant factor.
  template <bool UseCAS>
  bool Insert(const char* key, Splice* splice, bool allow_partial_splice_fix);

  // Returns true iff an entry that compares equal to key is in the list.
  bool Contains(const char* key) const;

  // Return estimated number of entries smaller than `key`.
  uint64_t EstimateCount(const char* key) const;

  // Validate correctness of the skip-list.
  void TEST_Validate() const;

  // Iteration over the contents of a skip list
  class Iterator {
   public:
    // Initialize an iterator over the specified list.
    // The returned iterator is not valid.
    explicit Iterator(const InlineSkipList* list);

    // Change the underlying skiplist used for this iterator
    // This enables us not changing the iterator without deallocating
    // an old one and then allocating a new one
    void SetList(const InlineSkipList* list);

    // Returns true iff the iterator is positioned at a valid node.
    bool Valid() const;

    // Returns the key at the current position.
    // REQUIRES: Valid()
    const char* key() const;

    // Advances to the next position.
    // REQUIRES: Valid()
    void Next();

    // Advances to the previous position.
    // REQUIRES: Valid()
    void Prev();

    // Advance to the first entry with a key >= target
    void Seek(const char* target);

    // Retreat to the last entry with a key <= target
    void SeekForPrev(const char* target);

    // Position at the first entry in list.
    // Final state of iterator is Valid() iff list is not empty.
    void SeekToFirst();

    // Position at the last entry in list.
    // Final state of iterator is Valid() iff list is not empty.
    void SeekToLast();

   private:
    const InlineSkipList* list_;
    Node* node_;
    // Intentionally copyable
  };

 private:
  const uint16_t kMaxHeight_;
  const uint16_t kBranching_;
  const uint32_t kScaledInverseBranching_;

  Allocator* const allocator_;  // Allocator used for allocations of nodes
  // Immutable after construction
  Comparator const compare_;
  Node* const head_;

  // Modified only by Insert().  Read racily by readers, but stale
  // values are ok.
  std::atomic<int> max_height_;  // Height of the entire list

  // seq_splice_ is a Splice used for insertions in the non-concurrent
  // case.  It caches the prev and next found during the most recent
  // non-concurrent insertion.
  Splice* seq_splice_;

  inline int GetMaxHeight() const {
    return max_height_.load(std::memory_order_relaxed);
  }

  int RandomHeight();

  Node* AllocateNode(size_t key_size, int height);

  bool Equal(const char* a, const char* b) const {
    return (compare_(a, b) == 0);
  }

  bool LessThan(const char* a, const char* b) const {
    return (compare_(a, b) < 0);
  }

  // Return true if key is greater than the data stored in "n".  Null n
  // is considered infinite.  n should not be head_.
  bool KeyIsAfterNode(const char* key, Node* n) const;
  bool KeyIsAfterNode(const DecodedKey& key, Node* n) const;

  // Returns the earliest node with a key >= key.
  // Return nullptr if there is no such node.
  Node* FindGreaterOrEqual(const char* key) const;

  // Return the latest node with a key < key.
  // Return head_ if there is no such node.
  // Fills prev[level] with pointer to previous node at "level" for every
  // level in [0..max_height_-1], if prev is non-null.
  Node* FindLessThan(const char* key, Node** prev = nullptr) const;

  // Return the latest node with a key < key on bottom_level. Start searching
  // from root node on the level below top_level.
  // Fills prev[level] with pointer to previous node at "level" for every
  // level in [bottom_level..top_level-1], if prev is non-null.
  Node* FindLessThan(const char* key, Node** prev, Node* root, int top_level,
                     int bottom_level) const;

  // Return the last node in the list.
  // Return head_ if list is empty.
  Node* FindLast() const;

  // Traverses a single level of the list, setting *out_prev to the last
  // node before the key and *out_next to the first node after. Assumes
  // that the key is not present in the skip list. On entry, before should
  // point to a node that is before the key, and after should point to
  // a node that is after the key.  after should be nullptr if a good after
  // node isn't conveniently available.
  template<bool prefetch_before>
  void FindSpliceForLevel(const DecodedKey& key, Node* before, Node* after, int level,
                          Node** out_prev, Node** out_next);

  // Recomputes Splice levels from highest_level (inclusive) down to
  // lowest_level (inclusive).
  void RecomputeSpliceLevels(const DecodedKey& key, Splice* splice,
                             int recompute_level);
};

// Implementation details follow

template <class Comparator>
struct InlineSkipList<Comparator>::Splice {
  // The invariant of a Splice is that prev_[i+1].key <= prev_[i].key <
  // next_[i].key <= next_[i+1].key for all i.  That means that if a
  // key is bracketed by prev_[i] and next_[i] then it is bracketed by
  // all higher levels.  It is _not_ required that prev_[i]->Next(i) ==
  // next_[i] (it probably did at some point in the past, but intervening
  // or concurrent operations might have inserted nodes in between).
  int height_ = 0;
  Node** prev_; // 数组首地址
  Node** next_;
};

// The Node data type is more of a pointer into custom-managed memory than
// a traditional C++ struct.  The key is stored in the bytes immediately
// after the struct, and the next_ pointers for nodes with height > 1 are
// stored immediately _before_ the struct.  This avoids the need to include
// any pointer or sizing data, which reduces per-node memory overheads.
template <class Comparator>
struct InlineSkipList<Comparator>::Node {
  // Stores the height of the node in the memory location normally used for
  // next_[0].  This is used for passing data from AllocateKey to Insert.
  void StashHeight(const int height) {
    assert(sizeof(int) <= sizeof(next_[0])); // next_[0] 存储 height， 一个int
    memcpy(static_cast<void*>(&next_[0]), &height, sizeof(int));
  }

  // Retrieves the value passed to StashHeight.  Undefined after a call
  // to SetNext or NoBarrier_SetNext.
  int UnstashHeight() const {
    int rv;
    memcpy(&rv, &next_[0], sizeof(int)); // 获取 next_[0]位置的int 值，一个heght
    return rv;
  }

  const char* Key() const { return reinterpret_cast<const char*>(&next_[1]); } // next_[1] 是顺序内存的下一个Node

  // Accessors/mutators for links.  Wrapped in methods so we can add
  // the appropriate barriers as necessary, and perform the necessary
  // addressing trickery for storing links below the Node in memory.
  Node* Next(int n) { // 获取下第n 层节点
    assert(n >= 0);
    // Use an 'acquire load' so that we observe a fully initialized
    // version of the returned Node.
    return ((&next_[0] - n)->load(std::memory_order_acquire));
  }

  void SetNext(int n, Node* x) { // 将 节点设置到 第n 层位置
    assert(n >= 0);
    // Use a 'release store' so that anybody who reads through this
    // pointer observes a fully initialized version of the inserted node.
    (&next_[0] - n)->store(x, std::memory_order_release);
  }

  bool CASNext(int n, Node* expected, Node* x) { // cas 获取 你下一个n位置node
    assert(n >= 0);
    return (&next_[0] - n)->compare_exchange_strong(expected, x);
  }

  // No-barrier variants that can be safely used in a few locations.
  Node* NoBarrier_Next(int n) {
    assert(n >= 0);
    return (&next_[0] - n)->load(std::memory_order_relaxed);
  }

  void NoBarrier_SetNext(int n, Node* x) {
    assert(n >= 0);
    (&next_[0] - n)->store(x, std::memory_order_relaxed);
  }

  // Insert node after prev on specific level.
  void InsertAfter(Node* prev, int level) {
    // NoBarrier_SetNext() suffices since we will add a barrier when
    // we publish a pointer to "this" in prev.
    NoBarrier_SetNext(level, prev->NoBarrier_Next(level));
    prev->SetNext(level, this); // 将node 设置到第level 层
  }

 private:
  // next_[0] is the lowest level link (level 0).  Higher levels are
  // stored _earlier_, so level 1 is at next_[-1].
  std::atomic<Node*> next_[1];
};

template <class Comparator>
inline InlineSkipList<Comparator>::Iterator::Iterator(
    const InlineSkipList* list) {
  SetList(list);
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::SetList(
    const InlineSkipList* list) {
  list_ = list;
  node_ = nullptr;
}

template <class Comparator>
inline bool InlineSkipList<Comparator>::Iterator::Valid() const {
  return node_ != nullptr;
}

template <class Comparator>
inline const char* InlineSkipList<Comparator>::Iterator::key() const {
  assert(Valid());
  return node_->Key();
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::Next() {
  assert(Valid());
  node_ = node_->Next(0);
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::Prev() {
  // Instead of using explicit "prev" links, we just search for the
  // last node that falls before key.
  assert(Valid());
  node_ = list_->FindLessThan(node_->Key());
  if (node_ == list_->head_) {
    node_ = nullptr;
  }
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::Seek(const char* target) {
  node_ = list_->FindGreaterOrEqual(target);
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::SeekForPrev(
    const char* target) {
  Seek(target);
  if (!Valid()) {
    SeekToLast();
  }
  while (Valid() && list_->LessThan(target, key())) {
    Prev();
  }
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::SeekToFirst() {
  node_ = list_->head_->Next(0);
}

template <class Comparator>
inline void InlineSkipList<Comparator>::Iterator::SeekToLast() {
  node_ = list_->FindLast();
  if (node_ == list_->head_) {
    node_ = nullptr;
  }
}

template <class Comparator>
int InlineSkipList<Comparator>::RandomHeight() { // 随机高度
  auto rnd = Random::GetTLSInstance();

  // Increase height with probability 1 in kBranching
  int height = 1;
  while (height < kMaxHeight_ && height < kMaxPossibleHeight &&
         rnd->Next() < kScaledInverseBranching_) {
    height++;
  }
  assert(height > 0);
  assert(height <= kMaxHeight_);
  assert(height <= kMaxPossibleHeight);
  return height;
}

template <class Comparator>
bool InlineSkipList<Comparator>::KeyIsAfterNode(const char* key,
                                                Node* n) const {
  // nullptr n is considered infinite
  assert(n != head_);
  return (n != nullptr) && (compare_(n->Key(), key) < 0); //  key 在右边， n 在左边
}

template <class Comparator>
bool InlineSkipList<Comparator>::KeyIsAfterNode(const DecodedKey& key,
                                                Node* n) const {
  // nullptr n is considered infinite
  assert(n != head_);
  return (n != nullptr) && (compare_(n->Key(), key) < 0);
}
// 寻找第一个大于或等于key 的节点
// 1   3   5
// 1 2 3 4 5   ,  find 4, 即找到4
template <class Comparator>
typename InlineSkipList<Comparator>::Node*
InlineSkipList<Comparator>::FindGreaterOrEqual(const char* key) const {
  // Note: It looks like we could reduce duplication by implementing
  // this function as FindLessThan(key)->Next(0), but we wouldn't be able
  // to exit early on equality and the result wouldn't even be correct.
  // A concurrent insert might occur after FindLessThan(key) but before
  // we get a chance to call Next(0).
  Node* x = head_; // 头
  int level = GetMaxHeight() - 1; // 最高层
  Node* last_bigger = nullptr;
  const DecodedKey key_decoded = compare_.decode_key(key);
  while (true) {
    Node* next = x->Next(level);
    if (next != nullptr) {
      PREFETCH(next->Next(level), 0, 1);
    }
    // Make sure the lists are sorted
    assert(x == head_ || next == nullptr || KeyIsAfterNode(next->Key(), x)); // next 在 x 右边
    // Make sure we haven't overshot during our search
    assert(x == head_ || KeyIsAfterNode(key_decoded, x));
    int cmp = (next == nullptr || next == last_bigger)
                  ? 1
                  : compare_(next->Key(), key_decoded); // next 在 key 的右边 --> 1
    if (cmp == 0 || (cmp > 0 && level == 0)) { // 如果是0层，则返回 next
      return next;
    } else if (cmp < 0) { // next 在key 的左边， --> -1,  链表递进
      // Keep searching in this list
      x = next;
    } else { // level 不为0 ，则层高降低，继续寻找
      // Switch to next list, reuse compare_() result
      last_bigger = next;
      level--;
    }
  }
}

template <class Comparator>
typename InlineSkipList<Comparator>::Node*
InlineSkipList<Comparator>::FindLessThan(const char* key, Node** prev) const {
  return FindLessThan(key, prev, head_, GetMaxHeight(), 0);
}

// 返回小于 key 的最后一个 Node
// 1   3   5
// 1 2 3 4 5   ,  find 4, 即找到 3
template <class Comparator>
typename InlineSkipList<Comparator>::Node*
InlineSkipList<Comparator>::FindLessThan(const char* key, Node** prev,
                                         Node* root, int top_level,
                                         int bottom_level) const {
  assert(top_level > bottom_level);
  int level = top_level - 1; // 最高层
  Node* x = root; // 根节点
  // KeyIsAfter(key, last_not_after) is definitely false
  Node* last_not_after = nullptr;
  const DecodedKey key_decoded = compare_.decode_key(key); // 编码
  while (true) {
    assert(x != nullptr);
    Node* next = x->Next(level); // level 层 next 节点
    if (next != nullptr) {
      PREFETCH(next->Next(level), 0, 1);
    }
    assert(x == head_ || next == nullptr || KeyIsAfterNode(next->Key(), x));
    assert(x == head_ || KeyIsAfterNode(key_decoded, x));
    if (next != last_not_after && KeyIsAfterNode(key_decoded, next)) { // key_decoded 在 next 右边
      // Keep searching in this list
      assert(next != nullptr);
      x = next; // 递进， x = next
    } else {  // 否则， key_decoded 不在 next 的右边 ，   key  next
      if (prev != nullptr) {             //           x      next
        prev[level] = x; // 设置prev                   pre    next
      }
      if (level == bottom_level) { // 如果level 是最底层， 则返回 x
        return x;
      } else { // 如果不是最底层， 则跳表降低一层，继续搜索
        // Switch to next list, reuse KeyIsAfterNode() result
        last_not_after = next; // 记录下一个节点， level 降低一次层
        level--;
      }
    }
  }
}
// 获取最后一个节点
// 1   3   5
// 1 2 3 4 5   ,  find, 即返回5
template <class Comparator>
typename InlineSkipList<Comparator>::Node*
InlineSkipList<Comparator>::FindLast() const {
  Node* x = head_; //head 节点
  int level = GetMaxHeight() - 1; // 最高层-1
  while (true) {
    Node* next = x->Next(level); // 遍历链表
    if (next == nullptr) { // 如果 next 节点为null
      if (level == 0) { // 如果是最底层，则返回 cur 节点
        return x;
      } else {
        // Switch to next list  如果不是最底层，则层高减去1
        level--;
      }
    } else { // 递进链表
      x = next;
    }
  }
}
// 查找跳表大约多少个元素
// 1   3   5
// 1 2 3 4 5   ,  即大于5个
template <class Comparator>
uint64_t InlineSkipList<Comparator>::EstimateCount(const char* key) const {
  uint64_t count = 0;

  Node* x = head_; // 头节点
  int level = GetMaxHeight() - 1; // 最大层高
  const DecodedKey key_decoded = compare_.decode_key(key); // key
  while (true) {
    assert(x == head_ || compare_(x->Key(), key_decoded) < 0);
    Node* next = x->Next(level); // 获取next
    if (next != nullptr) {
      PREFETCH(next->Next(level), 0, 1);
    }
    if (next == nullptr || compare_(next->Key(), key_decoded) >= 0) { // 如果next 在 key 的右边
      if (level == 0) { // 如果高度为0， 则返回count
        return count;
      } else {
        // Switch to next list
        count *= kBranching_; // 如果高度不为0， 则则count = count * branch， 分支因子
        level--; // 高度减1
      }
    } else { // 如果next 在key 左边，则递进链表
      x = next;
      count++;
    }
  }
}
// 初始化跳表
template <class Comparator>
InlineSkipList<Comparator>::InlineSkipList(const Comparator cmp,
                                           Allocator* allocator,
                                           int32_t max_height,
                                           int32_t branching_factor)
    : kMaxHeight_(static_cast<uint16_t>(max_height)), // 最大高度
      kBranching_(static_cast<uint16_t>(branching_factor)),
      kScaledInverseBranching_((Random::kMaxNext + 1) / kBranching_),
      allocator_(allocator), // 分配器
      compare_(cmp), // 比较器
      head_(AllocateNode(0, max_height)), // 头节点，头节点
      max_height_(1), // 最大高度
      seq_splice_(AllocateSplice()) {
  assert(max_height > 0 && kMaxHeight_ == static_cast<uint32_t>(max_height));
  assert(branching_factor > 1 &&
         kBranching_ == static_cast<uint32_t>(branching_factor));
  assert(kScaledInverseBranching_ > 0);

  for (int i = 0; i < kMaxHeight_; ++i) { // 遍历每层，
    head_->SetNext(i, nullptr);
  }
}

template <class Comparator>
char* InlineSkipList<Comparator>::AllocateKey(size_t key_size) {
  // 随机一个高度
  return const_cast<char*>(AllocateNode(key_size, RandomHeight())->Key());
}

/*
 *
 * Node {0,1,2,3,4} key
 * return Node[4]
 * */
template <class Comparator>
typename InlineSkipList<Comparator>::Node*
InlineSkipList<Comparator>::AllocateNode(size_t key_size, int height) {
  auto prefix = sizeof(std::atomic<Node*>) * (height - 1); // 高度为5， prefix 为4个Node 长度

  // prefix is space for the height - 1 pointers that we store before
  // the Node instance (next_[-(height - 1) .. -1]).  Node starts at
  // raw + prefix, and holds the bottom-mode (level 0) skip list pointer
  // next_[0].  key_size is the bytes for the key, which comes just after
  // the Node.
  // 高度-1 个指针。  //高度-1个Node指针，每个指针指向该高度的下一个节点
  char* raw = allocator_->AllocateAligned(prefix + sizeof(Node) + key_size);  // 4+1 Node +  ks(假如是4字节， 即将key 放在这里)
  Node* x = reinterpret_cast<Node*>(raw + prefix); // return 第[4] 个Node指针

  // Once we've linked the node into the skip list we don't actually need
  // to know its height, because we can implicitly use the fact that we
  // traversed into a node at level h to known that h is a valid level
  // for that node.  We need to convey the height to the Insert step,
  // however, so that it can perform the proper links.  Since we're not
  // using the pointers at the moment, StashHeight temporarily borrow
  // storage from next_[0] for that purpose.
  // //将节点高度暂时存储在高度为1的位置，插入完成后就不需要高度了，
  // //这个位置就会存放指向下一个节点的指针
  x->StashHeight(height);
  return x; // 返回一个node， node 私有成员有个 node数组
}

/*
 * [5 node*]  pre  node**
 * [4 node*]
 * [3 node*]
 * [2 node*]  next node**
 * [1 node*]
 * [0 node*]
 *        node
 *          |
 * Splice|node*|node*|node*|node*|node*|node*|
 * 数组里存储的是 node* 一级指针； 我指向某个位置的node*， 我就会是二级指针， node**
 * [->->][->] node
 * */
template <class Comparator>
typename InlineSkipList<Comparator>::Splice*
InlineSkipList<Comparator>::AllocateSplice() {
  // size of prev_ and next_
  size_t array_size = sizeof(Node*) * (kMaxHeight_ + 1); // kMaxHeight_ = 2, 则3个Node指针长度
  char* raw = allocator_->AllocateAligned(sizeof(Splice) + array_size * 2); // 分配6个 Node指针长度加一个 Splice长度
  Splice* splice = reinterpret_cast<Splice*>(raw); // Splice 头
  splice->height_ = 0;
  splice->prev_ = reinterpret_cast<Node**>(raw + sizeof(Splice));  // pre 是， Splice 头后的第一Node 指针
  splice->next_ = reinterpret_cast<Node**>(raw + sizeof(Splice) + array_size); // next 是第3个Node 指针
  return splice;
}

template <class Comparator>
typename InlineSkipList<Comparator>::Splice*
InlineSkipList<Comparator>::AllocateSpliceOnHeap() { // 分配在堆上
  size_t array_size = sizeof(Node*) * (kMaxHeight_ + 1);
  char* raw = new char[sizeof(Splice) + array_size * 2];
  Splice* splice = reinterpret_cast<Splice*>(raw);
  splice->height_ = 0;
  splice->prev_ = reinterpret_cast<Node**>(raw + sizeof(Splice));
  splice->next_ = reinterpret_cast<Node**>(raw + sizeof(Splice) + array_size);
  return splice;
}

template <class Comparator>
bool InlineSkipList<Comparator>::Insert(const char* key) {
  return Insert<false>(key, seq_splice_, false);
}

template <class Comparator>
bool InlineSkipList<Comparator>::InsertConcurrently(const char* key) { // 并发
  Node* prev[kMaxPossibleHeight];
  Node* next[kMaxPossibleHeight];
  Splice splice;
  splice.prev_ = prev;
  splice.next_ = next;
  return Insert<true>(key, &splice, false);
}

template <class Comparator>
bool InlineSkipList<Comparator>::InsertWithHint(const char* key, void** hint) { // 插入，有提示
  assert(hint != nullptr);
  Splice* splice = reinterpret_cast<Splice*>(*hint);
  if (splice == nullptr) {
    splice = AllocateSplice();
    *hint = reinterpret_cast<void*>(splice);
  }
  return Insert<false>(key, splice, true);
}

template <class Comparator>
bool InlineSkipList<Comparator>::InsertWithHintConcurrently(const char* key,
                                                            void** hint) {
  assert(hint != nullptr);
  Splice* splice = reinterpret_cast<Splice*>(*hint); // hint 获取splice
  if (splice == nullptr) {
    splice = AllocateSpliceOnHeap();
    *hint = reinterpret_cast<void*>(splice);
  }
  return Insert<true>(key, splice, true); // 插入key
}

//                      |       |
// 1      2      3      4      4.5       5      6   找4
// pre   after
// 返回          pre     next
// 返回                  pre             next
template <class Comparator>
template <bool prefetch_before>
void InlineSkipList<Comparator>::FindSpliceForLevel(const DecodedKey& key,
                                                    Node* before, Node* after,
                                                    int level, Node** out_prev,
                                                    Node** out_next) {
  while (true) {
    Node* next = before->Next(level); // 获取befor 节点第levle 层 next 节点
    if (next != nullptr) { // 不为null， 则prefetch， 预取 level层
      PREFETCH(next->Next(level), 0, 1);
    }
    if (prefetch_before == true) {
      if (next != nullptr && level>0) { // 预取 level-1 层
        PREFETCH(next->Next(level-1), 0, 1);
      }
    }
    assert(before == head_ || next == nullptr ||
           KeyIsAfterNode(next->Key(), before));  // next->Key() 在befor之后
    assert(before == head_ || KeyIsAfterNode(key, before));
    if (next == after || !KeyIsAfterNode(key, next)) { // key在next 之前， 则break
      // found it
      *out_prev = before;
      *out_next = next;
      return;
    }
    before = next; // befor 后移
  }
}

/*
 *
 * 开始从 splice->prev_[max] 即跳表 [max] 层的头结点遍历找到 key 的pre 和 next 指针，放在 splice->prev_[max-1] 位置
 * 然后从 splice->prev_[max-1] 即跳表[max-1] 的起始位置，开始搜索得到 pre 和 next 指针， 结果放在 splice->prev_[max-2] 位置
 *
 * 3                        18
 * 3    4     10            18
 * 3    4  7  20     11     18
 * key 为11
 * 从第[2]层找到  pre = 3, next = 18
 * 从第[1]层找到  pre = 10, next = 18
 * 从第[0]层找到  pre = 20, next = 11
 * */
template <class Comparator>
void InlineSkipList<Comparator>::RecomputeSpliceLevels(const DecodedKey& key,
                                                       Splice* splice,
                                                       int recompute_level) {
  assert(recompute_level > 0);
  assert(recompute_level <= splice->height_);
  for (int i = recompute_level - 1; i >= 0; --i) { // 遍历 recompute_level 之下的所有层， 从下一层找到 pre 和 next，放在本层位置
    FindSpliceForLevel<true>(key, splice->prev_[i + 1], splice->next_[i + 1], i,
                       &splice->prev_[i], &splice->next_[i]);
  }
}

template <class Comparator>
template <bool UseCAS>
bool InlineSkipList<Comparator>::Insert(const char* key, Splice* splice,
                                        bool allow_partial_splice_fix) {
  Node* x = reinterpret_cast<Node*>(const_cast<char*>(key)) - 1;  // key 的前一个node, 这里没看懂
  const DecodedKey key_decoded = compare_.decode_key(key);
  int height = x->UnstashHeight(); // 获取 height
  assert(height >= 1 && height <= kMaxHeight_);

  int max_height = max_height_.load(std::memory_order_relaxed); // 最高层
  while (height > max_height) { // 如果大于最高层， 更新最高层
    if (max_height_.compare_exchange_weak(max_height, height)) {
      // successfully updated it
      max_height = height;
      break;
    }
    // else retry, possibly exiting the loop because somebody else
    // increased it
  }
  assert(max_height <= kMaxPossibleHeight);

  int recompute_height = 0;
  if (splice->height_ < max_height) {
    // Either splice has never been used or max_height has grown since
    // last use.  We could potentially fix it in the latter case, but
    // that is tricky.
    splice->prev_[max_height] = head_; // 最高层， pre 为head   0 层的头结点
    splice->next_[max_height] = nullptr; // 最高层，next 为 null
    splice->height_ = max_height; // 最高层
    recompute_height = max_height;
  } else {
    // Splice is a valid proper-height splice that brackets some
    // key, but does it bracket this one?  We need to validate it and
    // recompute a portion of the splice (levels 0..recompute_height-1)
    // that is a superset of all levels that don't bracket the new key.
    // Several choices are reasonable, because we have to balance the work
    // saved against the extra comparisons required to validate the Splice.
    //
    // One strategy is just to recompute all of orig_splice_height if the
    // bottom level isn't bracketing.  This pessimistically assumes that
    // we will either get a perfect Splice hit (increasing sequential
    // inserts) or have no locality.
    //
    // Another strategy is to walk up the Splice's levels until we find
    // a level that brackets the key.  This strategy lets the Splice
    // hint help for other cases: it turns insertion from O(log N) into
    // O(log D), where D is the number of nodes in between the key that
    // produced the Splice and the current insert (insertion is aided
    // whether the new key is before or after the splice).  If you have
    // a way of using a prefix of the key to map directly to the closest
    // Splice out of O(sqrt(N)) Splices and we make it so that splices
    // can also be used as hints during read, then we end up with Oshman's
    // and Shavit's SkipTrie, which has O(log log N) lookup and insertion
    // (compare to O(log N) for skip list).
    //
    // We control the pessimistic strategy with allow_partial_splice_fix.
    // A good strategy is probably to be pessimistic for seq_splice_,
    // optimistic if the caller actually went to the work of providing
    // a Splice.
    while (recompute_height < max_height) {
      if (splice->prev_[recompute_height]->Next(recompute_height) !=
          splice->next_[recompute_height]) {
        // splice isn't tight at this level, there must have been some inserts
        // to this
        // location that didn't update the splice.  We might only be a little
        // stale, but if
        // the splice is very stale it would be O(N) to fix it.  We haven't used
        // up any of
        // our budget of comparisons, so always move up even if we are
        // pessimistic about
        // our chances of success.
        ++recompute_height;
      } else if (splice->prev_[recompute_height] != head_ &&
                 !KeyIsAfterNode(key_decoded,
                                 splice->prev_[recompute_height])) {
        // key is from before splice
        if (allow_partial_splice_fix) {
          // skip all levels with the same node without more comparisons
          Node* bad = splice->prev_[recompute_height];
          while (splice->prev_[recompute_height] == bad) {
            ++recompute_height;
          }
        } else {
          // we're pessimistic, recompute everything
          recompute_height = max_height;
        }
      } else if (KeyIsAfterNode(key_decoded,
                                splice->next_[recompute_height])) {
        // key is from after splice
        if (allow_partial_splice_fix) {
          Node* bad = splice->next_[recompute_height];
          while (splice->next_[recompute_height] == bad) {
            ++recompute_height;
          }
        } else {
          recompute_height = max_height;
        }
      } else {
        // this level brackets the key, we won!
        break;
      }
    }
  }
  assert(recompute_height <= max_height);
  if (recompute_height > 0) {
    RecomputeSpliceLevels(key_decoded, splice, recompute_height);
  }

  bool splice_is_valid = true;
  if (UseCAS) {
    for (int i = 0; i < height; ++i) { // 从0层 到 最高层
      while (true) {
        // Checking for duplicate keys on the level 0 is sufficient  检查重复键
        if (UNLIKELY(i == 0 && splice->next_[i] != nullptr &&
                     compare_(x->Key(), splice->next_[i]->Key()) >= 0)) { // 第0层，x 的key 大于或等于 splice.next key
          // duplicate key
          return false;
        }
        if (UNLIKELY(i == 0 && splice->prev_[i] != head_ &&
                     compare_(splice->prev_[i]->Key(), x->Key()) >= 0)) { // 第0 层， pre key 大于或等于x 的key
          // duplicate key
          return false;
        }
        assert(splice->next_[i] == nullptr ||
               compare_(x->Key(), splice->next_[i]->Key()) < 0);
        assert(splice->prev_[i] == head_ ||
               compare_(splice->prev_[i]->Key(), x->Key()) < 0);   // key 是小于 next， 大于 pre；  pre < key < next
        x->NoBarrier_SetNext(i, splice->next_[i]);   // x 设置next指针， splice->next_[i] ， splice->prev_[i] 节点设置next指针指向x
        if (splice->prev_[i]->CASNext(i, splice->next_[i], x)) { // 插入节点 x ， 本层插入成功，下一层插入
          // success
          break;
        }
        // CAS failed, we need to recompute prev and next. It is unlikely
        // to be helpful to try to use a different level as we redo the
        // search, because it should be unlikely that lots of nodes have
        // been inserted between prev[i] and next[i]. No point in using
        // next[i] as the after hint, because we know it is stale.
        FindSpliceForLevel<false>(key_decoded, splice->prev_[i], nullptr, i,
                                  &splice->prev_[i], &splice->next_[i]);

        // Since we've narrowed the bracket for level i, we might have
        // violated the Splice constraint between i and i-1.  Make sure
        // we recompute the whole thing next time.
        if (i > 0) {
          splice_is_valid = false;
        }
      }
    }
  } else {
    for (int i = 0; i < height; ++i) {
      if (i >= recompute_height &&
          splice->prev_[i]->Next(i) != splice->next_[i]) {
        FindSpliceForLevel<false>(key_decoded, splice->prev_[i], nullptr, i,
                                  &splice->prev_[i], &splice->next_[i]);
      }
      // Checking for duplicate keys on the level 0 is sufficient
      if (UNLIKELY(i == 0 && splice->next_[i] != nullptr &&
                   compare_(x->Key(), splice->next_[i]->Key()) >= 0)) {
        // duplicate key
        return false;
      }
      if (UNLIKELY(i == 0 && splice->prev_[i] != head_ &&
                   compare_(splice->prev_[i]->Key(), x->Key()) >= 0)) {
        // duplicate key
        return false;
      }
      assert(splice->next_[i] == nullptr ||
             compare_(x->Key(), splice->next_[i]->Key()) < 0);
      assert(splice->prev_[i] == head_ ||
             compare_(splice->prev_[i]->Key(), x->Key()) < 0);
      assert(splice->prev_[i]->Next(i) == splice->next_[i]);
      x->NoBarrier_SetNext(i, splice->next_[i]);
      splice->prev_[i]->SetNext(i, x);
    }
  }
  if (splice_is_valid) {
    for (int i = 0; i < height; ++i) {
      splice->prev_[i] = x;
    }
    assert(splice->prev_[splice->height_] == head_);
    assert(splice->next_[splice->height_] == nullptr);
    for (int i = 0; i < splice->height_; ++i) {
      assert(splice->next_[i] == nullptr ||
             compare_(key, splice->next_[i]->Key()) < 0);
      assert(splice->prev_[i] == head_ ||
             compare_(splice->prev_[i]->Key(), key) <= 0);
      assert(splice->prev_[i + 1] == splice->prev_[i] ||
             splice->prev_[i + 1] == head_ ||
             compare_(splice->prev_[i + 1]->Key(), splice->prev_[i]->Key()) <
                 0);
      assert(splice->next_[i + 1] == splice->next_[i] ||
             splice->next_[i + 1] == nullptr ||
             compare_(splice->next_[i]->Key(), splice->next_[i + 1]->Key()) <
                 0);
    }
  } else {
    splice->height_ = 0;
  }
  return true;
}

// 返回跳表是否包含该key
template <class Comparator>
bool InlineSkipList<Comparator>::Contains(const char* key) const {
  Node* x = FindGreaterOrEqual(key); // 找大于或等于key 的第一个节点
  if (x != nullptr && Equal(key, x->Key())) { // 如果节点就是key，包含该key
    return true;
  } else {
    return false;
  }
}

template <class Comparator>
void InlineSkipList<Comparator>::TEST_Validate() const {
  // Interate over all levels at the same time, and verify nodes appear in
  // the right order, and nodes appear in upper level also appear in lower
  // levels.
  Node* nodes[kMaxPossibleHeight];
  int max_height = GetMaxHeight();
  assert(max_height > 0);
  for (int i = 0; i < max_height; i++) {
    nodes[i] = head_;
  }
  while (nodes[0] != nullptr) {
    Node* l0_next = nodes[0]->Next(0);
    if (l0_next == nullptr) {
      break;
    }
    assert(nodes[0] == head_ || compare_(nodes[0]->Key(), l0_next->Key()) < 0);
    nodes[0] = l0_next;

    int i = 1;
    while (i < max_height) {
      Node* next = nodes[i]->Next(i);
      if (next == nullptr) {
        break;
      }
      auto cmp = compare_(nodes[0]->Key(), next->Key());
      assert(cmp <= 0);
      if (cmp == 0) {
        assert(next == nodes[0]);
        nodes[i] = next;
      } else {
        break;
      }
      i++;
    }
  }
  for (int i = 1; i < max_height; i++) {
    assert(nodes[i] != nullptr && nodes[i]->Next(i) == nullptr);
  }
}

}  // namespace ROCKSDB_NAMESPACE
