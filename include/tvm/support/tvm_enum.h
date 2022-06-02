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
 *
 * \file tvm_enum.h
 * \brief Macro support function
 */
#ifndef TVM_SUPPORT_TVM_ENUM_H_
#define TVM_SUPPORT_TVM_ENUM_H_

#include <tvm/runtime/logging.h>

#include <string>

#include "macro_foreach.h"

namespace tvm {
namespace support {

// A couple utilities to defer evaluation of a macro.
#define TVM_MACRO_ESCAPE(...) TVM_MACRO_ESCAPE_(__VA_ARGS__)
#define TVM_MACRO_ESCAPE_(...) TVM_MACRO_REMOVE_##__VA_ARGS__

// Extract the enum name.  When X is a name, it generates just that
// name.  When X is a (name,value) pair, it returns the first item of
// that pair.  This uses the same trick shown
// https://stackoverflow.com/a/62984543/2689797 to conditionally
// execute a macro.
#define TVM_ENUM_NAME(X) TVM_MACRO_ESCAPE(TVM_MACRO_DEFER_ENUM_NAME X)
#define TVM_MACRO_DEFER_ENUM_NAME(name, value) TVM_MACRO_DEFER_ENUM_NAME name
#define TVM_MACRO_REMOVE_TVM_MACRO_DEFER_ENUM_NAME

#define TVM_ENUM_STRING_NAME(item) TVM_MACRO_STRINGIZE(TVM_ENUM_NAME(item))
#define TVM_MACRO_STRINGIZE(x) TVM_MACRO_STRINGIZE_IMPL(x)
#define TVM_MACRO_STRINGIZE_IMPL(x) #x

// Make the enum declaration.  When X is a name, it generates just
// that name.  When X is a (name,value) pair, it generates name=value.
// This uses the same trick shown
// https://stackoverflow.com/a/62984543/2689797 to conditionally
// execute a macro.
#define TVM_ENUM_DECL_ITEM(X) TVM_MACRO_ESCAPE(TVM_MACRO_DEFER_ENUM_DECL X)
#define TVM_MACRO_DEFER_ENUM_DECL(name, value) TVM_MACRO_DEFER_ENUM_DECL name = value
#define TVM_MACRO_REMOVE_TVM_MACRO_DEFER_ENUM_DECL

#define TVM_ENUM_HOIST_ITEM(item) \
  static constexpr auto TVM_ENUM_NAME(item) = Value::TVM_ENUM_NAME(item);

#define TVM_ENUM_TOSTRING_ITEM(item) \
  case TVM_ENUM_NAME(item):          \
    return TVM_ENUM_STRING_NAME(item);

#define TVM_ENUM_FROMSTRING_ITEM(item)     \
  if (val == TVM_ENUM_STRING_NAME(item)) { \
    val_ = Value::TVM_ENUM_NAME(item);     \
  } else

#define TVM_MACRO_INCREMENT(...) +1

#define TVM_ENUM(EnumName, ...)                                                     \
  class EnumName : public ::tvm::support::EnumCRTP<EnumName> {                      \
   public:                                                                          \
    /* Declare the enum, using any specific values provided in the definition */    \
    enum class Value : int { MAP_LIST(TVM_ENUM_DECL_ITEM, __VA_ARGS__) };           \
                                                                                    \
    /* Hoist the enum values, to allow EnumName::kEnum                              \
     * instead of EnumName::Value::kEnum                                            \
     */                                                                             \
    MAP(TVM_ENUM_HOIST_ITEM, __VA_ARGS__)                                           \
                                                                                    \
    /* Explicit construct from int */                                               \
    explicit EnumName(int val) : val_(Value(val)) {}                                \
    /* Implicit construct from enum class, in case hoisted values are used */       \
    EnumName(Value val) : val_(val) {}                                              \
    explicit EnumName(std::string val) {                                            \
      MAP(TVM_ENUM_FROMSTRING_ITEM, __VA_ARGS__) {                                  \
        LOG(FATAL) << "Cannot construct " #EnumName " from value " << val;          \
      }                                                                             \
    }                                                                               \
                                                                                    \
    std::string ToString() const {                                                  \
      switch (val_) {                                                               \
        MAP(TVM_ENUM_TOSTRING_ITEM, __VA_ARGS__)                                    \
        default:                                                                    \
          LOG(FATAL) << "Value " << int(val_)                                       \
                     << " does not correspond to any named value in " #EnumName;    \
          return "";                                                                \
      }                                                                             \
    }                                                                               \
                                                                                    \
    bool operator==(const EnumName& other) const { return val_ == other.val_; }     \
    bool operator!=(const EnumName& other) const { return !(*this == other); }      \
                                                                                    \
   private:                                                                         \
    Value val_;                                                                     \
                                                                                    \
    static const size_t n_values = 0 MAP(TVM_MACRO_INCREMENT, __VA_ARGS__);         \
    static const constexpr Value values[] = {MAP_LIST(TVM_ENUM_NAME, __VA_ARGS__)}; \
    static const constexpr char* const value_names[] = {                            \
        MAP_LIST(TVM_ENUM_STRING_NAME, __VA_ARGS__)};                               \
    static const constexpr char* const enum_name = #EnumName;                       \
  }

template <typename T>
class EnumCRTP {
 public:
  // TODO: Add the FFI interface here
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_TVM_ENUM_H_
