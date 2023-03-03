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
 * \file stmt_to_primfunc.h
 *
 * \brief Analyze a tir::Stmt to generate a PrimFunc with that body
 */

#ifndef TVM_TIR_ANALYSIS_STMT_TO_PRIMFUNC_H_
#define TVM_TIR_ANALYSIS_STMT_TO_PRIMFUNC_H_

#include <tvm/tir/function.h>
#include <tvm/relax/expr.h>

#include <tuple>

namespace tvm {
namespace tir {

/* \brief Utility class with the wrapped body
 *
 * Methods in this class convert the result into the context-dependent types.
 */
class StmtToPrimFuncResult {
public:
  enum class BufferParameter {
    // Represent tir::Buffer parameters as void* for static shapes,
    // and as DLTensor for dynamic shapes.  This avoids introducing
    // additional tir::Var parameters.  The resulting
    // PrimFunc::buffer_map will contain an entry for each
    // dynamic-shaped buffer.
    //
    // AutoSelect,

    // Represent tir::Buffer parameters as DLTensor.  The
    // PrimFunc::buffer_map will contain an entry for each buffer.
    DLTensor,

    // Represent tir::Buffer parameters as void*.  The
    // PrimFunc::buffer_map will be empty.
    //
    // VoidPtr,
  };

  StmtToPrimFuncResult(Stmt body, std::vector<Buffer> input_buffer_params, std::vector<Buffer> output_buffer_params,
                       std::vector<Var> var_params, std::vector<Var> implicit_var_defs)
    : body_(std::move(body)), input_buffer_params_(std::move(input_buffer_params)),
      output_buffer_params_(std::move(output_buffer_params)),
      var_params_(var_params), implicit_var_defs_(implicit_var_defs) {}

  PrimFunc ToPrimFunc(BufferParameter conv = BufferParameter::DLTensor) const;
  Array<PrimExpr> TIRCallSiteArgs(BufferParameter conv = BufferParameter::DLTensor) const;

private:
  Stmt body_;
  std::vector<Buffer> input_buffer_params_;
  std::vector<Buffer> output_buffer_params_;
  std::vector<Var> var_params_;
  std::vector<Var> implicit_var_defs_;
};

/* \brief Generate a PrimFunc with the given body
 *
 * \param body The body of the new PrimFunc.  The body is assumed to
 * be well-formed, with SSA and no undefined variables, at the context
 * in which it appears.
 *
 * \returns The generated PrimFunc, and the arguments that should be
 * provided at the call-site.
 */
StmtToPrimFuncResult StmtToPrimFunc(Stmt body);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_ANALYSIS_STMT_TO_PRIMFUNC_H_
