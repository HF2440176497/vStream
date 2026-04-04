/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef MODULES_INFERENCE_HPP_
#define MODULES_INFERENCE_HPP_


#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cnstream_module.hpp"
#include "exception.hpp"

#define DECLARE_PRIVATE(d_ptr, Class) \
  friend class Class##Private;        \
  Class##Private* d_ptr = nullptr;

#define DECLARE_PUBLIC(q_ptr, Class) \
  friend class Class;                \
  Class* q_ptr = nullptr;

namespace cnstream {

CNSTREAM_REGISTER_EXCEPTION(Inference);

class InferencePrivate;
class InferParamManager;


class Inference : public Module, public ModuleCreator<Inference> {
 public:
  /**
   * @brief Creates Inference module.
   *
   * @param[in] name The name of the Inference module.
   *
   * @return None.
   */
  explicit Inference(const std::string& name);
  /**
   * @brief Destructor, destructs the inference instance.
   *
   * @param None.
   *
   * @return None.
   */
  virtual ~Inference();

  bool Open(ModuleParamSet paramSet) override;
  void Close() override;
  int  Process(std::shared_ptr<FrameInfo> data) final;
  bool CheckParamSet(const ModuleParamSet& param_set) const override;

 private:
  std::shared_ptr<InferParamManager> param_manager_ = nullptr;
  DECLARE_PRIVATE(d_ptr_, Inference);
};  // class Inference

REGISTER_MODULE(Inference);

}  // namespace cnstream


#endif  // MODULES_INFERENCE_HPP_
