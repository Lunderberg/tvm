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

#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <variant>

namespace tvm {
namespace support {

class DebugTimer {
 public:
  using clock = std::chrono::steady_clock;

  DebugTimer(std::string name) : name(name) {}

  DebugTimer& subcategory(std::string subcategory) {
    opt_subcategory = subcategory;
    return *this;
  }

  DebugTimer& subcategory(std::function<void(std::ostream&)> func) {
    std::stringstream ss;
    func(ss);
    opt_subcategory = ss.str();
    return *this;
  }

  DebugTimer& ms_required_to_print(int ms) {
    minimum_print_time = std::chrono::milliseconds(ms);
    return *this;
  }

  DebugTimer& always_print_short_timers(bool always_print = true) {
    if (always_print) {
      minimum_print_time = std::nullopt;
    } else if (!minimum_print_time.has_value()) {
      minimum_print_time = std::chrono::milliseconds(5);
    }
    return *this;
  }

  DebugTimer& on_start(std::function<void(std::ostream&)> func) {
    details_on_start = func;
    return *this;
  }

  DebugTimer& on_start(std::string message) {
    details_on_start = [message](std::ostream& out) { out << message; };
    return *this;
  }

  DebugTimer& on_finish(std::function<void(std::ostream&)> func) {
    details_on_finish = func;
    return *this;
  }

  DebugTimer& on_finish(std::string message) {
    details_on_finish = [message = std::move(message)](std::ostream& out) { out << message; };
    return *this;
  }

  std::string name;
  std::optional<std::string> opt_subcategory = std::nullopt;
  std::optional<std::chrono::milliseconds> minimum_print_time = std::chrono::milliseconds(5);
  std::function<void(std::ostream&)> details_on_start = nullptr;
  std::function<void(std::ostream&)> details_on_finish = nullptr;

  class Impl {
    using clock = std::chrono::steady_clock;

   public:
    Impl(DebugTimer builder)
        : Impl(builder.name, builder.opt_subcategory, builder.minimum_print_time,
               builder.details_on_start, builder.details_on_finish) {}

   private:
    Impl(std::string name, std::optional<std::string> subcategory,
         std::optional<std::chrono::milliseconds> minimum_print_time,
         std::function<void(std::ostream&)> details_on_start,
         std::function<void(std::ostream&)> details_on_finish)
        : name_(name),
          subcategory_(subcategory),
          minimum_print_time_(minimum_print_time),
          details_on_start_(details_on_start),
          details_on_finish_(details_on_finish),
          nested_depth(current_nested_depth()),
          parent_timer(innermost_timer()),
          start_(clock::now()) {
      if (minimum_print_time_.has_value()) {
        cache_start_message();
      } else {
        print_start_message();
      }

      if (parent_timer) {
        assert(nested_depth == parent_timer->nested_depth + 1);
      }

      active_timers().push_back(this);
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

   public:
    ~Impl() {
      assert(active_timers.size() && active_timers().back() == this);

      active_timers().pop_back();

      auto end = clock::now();
      auto duration = end - start_;

      if (!minimum_print_time_.has_value() || duration >= *minimum_print_time_ ||
          std::get_if<std::monostate>(&details_on_start_)) {
        if (parent_timer) {
          parent_timer->print_start_message();
        }
        auto duration_ms = duration / std::chrono::milliseconds(1);
        std::cout << std::string(4 * nested_depth, ' ') << "Finished " << name_;
        if (subcategory_.has_value()) {
          std::cout << " " << *subcategory_;
        }
        std::cout << " after " << duration_ms << " ms";

        if (details_on_finish_) {
          std::cout << ", ";
          details_on_finish_(std::cout);
        }
        std::cout << std::endl;
      }
    }

   private:
    void cache_start_message() {
      if (auto* as_func = std::get_if<std::function<void(std::ostream&)>>(&details_on_start_)) {
        if (*as_func) {
          std::stringstream ss;
          (*as_func)(ss);
          details_on_start_ = ss.str();
        } else {
          details_on_start_ = std::string();
        }
      }
    }

    void print_start_message() {
      if (std::get_if<std::monostate>(&details_on_start_)) {
        return;
      }

      if (parent_timer) {
        parent_timer->print_start_message();
      }

      std::cout << std::string(4 * nested_depth, ' ');

      if (std::get_if<std::string>(&details_on_start_)) {
        std::cout << "(Out-of-order print) ";
      }

      std::cout << "Started " << name_;
      if (subcategory_.has_value()) {
        std::cout << " " << *subcategory_;
      }

      if (auto* as_func = std::get_if<std::function<void(std::ostream&)>>(&details_on_start_);
          as_func && *as_func) {
        std::cout << ", ";
        (*as_func)(std::cout);
      } else if (auto* as_string = std::get_if<std::string>(&details_on_start_)) {
        std::cout << ", ";
        std::cout << *as_string;
      }

      std::cout << std::endl;

      details_on_start_ = std::monostate();
    }

    static std::vector<Impl*>& active_timers() {
      static std::vector<Impl*> active_timers;
      return active_timers;
    }
    static size_t current_nested_depth() { return active_timers().size(); }

    static Impl* innermost_timer() {
      const auto& active_timers = Impl::active_timers();
      if (active_timers.empty()) {
        return nullptr;
      } else {
        return active_timers.back();
      }
    }

    friend class DebugTimer;
    friend class std::optional<Impl>;

    // User-provided parameters
    std::string name_;
    std::optional<std::string> subcategory_;
    std::optional<std::chrono::milliseconds> minimum_print_time_ = std::nullopt;
    std::variant<std::function<void(std::ostream&)>, std::string, std::monostate> details_on_start_;
    std::function<void(std::ostream&)> details_on_finish_;

    size_t nested_depth = 0;
    Impl* parent_timer = nullptr;

    clock::time_point start_;
  };

  [[nodiscard]] Impl start() {
    return Impl(name, opt_subcategory, minimum_print_time, details_on_start, details_on_finish);
  }

  void start(std::optional<Impl>& opt) {
    opt.reset();
    opt.emplace(*this);
  }
};

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_DEBUG_TIMER_H_
