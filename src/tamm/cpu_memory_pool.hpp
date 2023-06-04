#pragma once

#include <cstddef>
#include <new>
#include <unordered_map>
#include <vector>
#if __APPLE__
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

namespace tamm {

class CPUPooledStorageManager {
protected:
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;

private:
  CPUPooledStorageManager() { reserve_ = 90; }
  ~CPUPooledStorageManager() { ReleaseAll(); }

public:
  void* allocate(size_t sizeInBytes) {
    // don't allocate anything if the user requested zero bytes
    if(0 == sizeInBytes) { return nullptr; }
    auto&& reuse_it = memory_pool_.find(sizeInBytes);
    if(reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
      size_t free{}, total{};

      struct sysinfo cpumeminfo_;
      sysinfo(&cpumeminfo_);
      total = cpumeminfo_.totalram * cpumeminfo_.mem_unit;
      free  = cpumeminfo_.freeram * cpumeminfo_.mem_unit;

      if(free <= total * reserve_ / 100 || sizeInBytes > free - total * reserve_ / 100) {
        ReleaseAll();
      }

      void* ret = ::operator new(sizeInBytes);

      used_memory_ += sizeInBytes;
      return ret;
    }
    else {
      auto&& reuse_pool = reuse_it->second;
      auto   ret        = reuse_pool.back();
      reuse_pool.pop_back();
      return ret;
    }
  }
  void deallocate(void* ptr, size_t sizeInBytes) {
    auto&& reuse_pool = memory_pool_[sizeInBytes];
    reuse_pool.push_back(ptr);
  }

  // void cpuMemset(void*& ptr, size_t sizeInBytes, bool blocking = false) {}

  void ReleaseAll() {
    for(auto&& i: memory_pool_) {
      for(auto&& j: i.second) {
        ::operator delete(j);
        used_memory_ -= i.first;
      }
    }
    memory_pool_.clear();
  }

  /// Returns the instance of device manager singleton.
  inline static CPUPooledStorageManager& getInstance() {
    static CPUPooledStorageManager d_m{};
    return d_m;
  }

  CPUPooledStorageManager(const CPUPooledStorageManager&)            = delete;
  CPUPooledStorageManager& operator=(const CPUPooledStorageManager&) = delete;
  CPUPooledStorageManager(CPUPooledStorageManager&&)                 = delete;
  CPUPooledStorageManager& operator=(CPUPooledStorageManager&&)      = delete;

}; // class CPUPooledStorageManager

} // namespace tamm
