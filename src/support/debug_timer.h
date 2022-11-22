/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file debug_timer.h
 * \brief Defines some common utility function..
 */
#ifndef TVM_SUPPORT_DEBUG_TIMER_H_
#define TVM_SUPPORT_DEBUG_TIMER_H_

#include <chrono>
#include <iostream>

namespace tvm {
namespace support {

class DebugTimer {
 public:
  using clock = std::chrono::high_resolution_clock;

  DebugTimer(std::string activity, int min_print_ms = 0)
      : activity_(activity), min_print_ms_(min_print_ms) {
    if (min_print_ms_ == 0) {
      std::cout << "Started " << activity_ << std::endl;
    }
    start_ = clock::now();
  }
  ~DebugTimer() {
    auto end = clock::now();
    auto duration = end - start_;
    auto duration_ms = duration / std::chrono::milliseconds(1);
    if (min_print_ms_ == 0 || duration_ms >= min_print_ms_) {
      std::cout << "Finished " << activity_ << " after " << duration_ms << " ms" << std::endl;
    }
  }

 private:
  std::string activity_;
  int min_print_ms_;
  clock::time_point start_;
};

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_DEBUG_TIMER_H_
