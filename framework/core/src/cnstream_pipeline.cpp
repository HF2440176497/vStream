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

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_graph.hpp"
#include "cnstream_module.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_connector.hpp"
#include "cnstream_conveyor.hpp"

#include "util/cnstream_queue.hpp"

namespace cnstream {

Pipeline::Pipeline(const std::string& name) : name_(name) {
  // stream message handle thread
  exit_msg_loop_ = false;
  smsg_thread_ = std::thread(&Pipeline::StreamMsgHandleFunc, this);

  event_bus_.reset(new (std::nothrow) EventBus());
  LOGF_IF(CORE, nullptr == event_bus_) << "Pipeline::Pipeline() failed to alloc EventBus";
  GetEventBus()->AddBusWatch(std::bind(&Pipeline::DefaultBusWatch, this, std::placeholders::_1));

  idxManager_.reset(new (std::nothrow) IdxManager());
  LOGF_IF(CORE, nullptr == idxManager_) << "Pipeline::Pipeline() failed to alloc IdxManager";

  graph_.reset(new (std::nothrow) CNGraph<NodeContext>());
  LOGF_IF(CORE, nullptr == graph_) << "Pipeline::Pipeline() failed to alloc CNGraph";
}

Pipeline::~Pipeline() {
  Stop();
  exit_msg_loop_ = true;
  if (smsg_thread_.joinable()) {
    smsg_thread_.join();
  }
  event_bus_.reset();
  graph_.reset();  // must release before idxManager_;
  idxManager_.reset();
}

bool Pipeline::BuildPipeline(const CNGraphConfig& graph_config) {
  auto t = graph_config;
  t.name = GetName();
  if (!graph_->Init(t)) {
    LOGE(CORE) << "Init graph failed.";
    return false;
  }
  // create modules by config
  std::vector<std::shared_ptr<Module>> modules;  // used to init profiler
  if (!CreateModules(&modules)) {
    LOGE(CORE) << "Create modules failed.";
    return false;
  }

  // generate parant mask for all nodes and route mask for head nodes.
  GenerateModulesMask();

  // generate module profilers
  GenerateModulesProfilers(modules);

  // create connectors for all nodes beside head nodes. This call must after GenerateModulesMask called,
  return CreateConnectors();
}

bool Pipeline::IsProfilingEnabled() const {
  if (nullptr == graph_) {
    LOGF(CORE) << "Pipeline::IsProfilingEnabled() failed, graph_ is null.";
    return false;
  }
  if (graph_->GetConfig().profiler_config.enable_profile) {
    return true;
  }
  return false;
}

/**
 * @brief 为每个 Module 创建一个 ModuleProfiler 都注册一个 kPROCESS_PROFILER_NAME
 */
void Pipeline::GenerateModulesProfilers(const std::vector<std::shared_ptr<Module>>& modules) {
  for (const auto& module: modules) {
    auto name = module->GetName();
    module_profilers_.emplace(name, std::unique_ptr<ModuleProfiler>(new ModuleProfiler(graph_->GetConfig().profiler_config, name)));
    module_profilers_[name]->RegisterProcess(kPROCESS_PROFILER_NAME);
  }
}

// note: dereference unique_ptr
ModuleProfiler* Pipeline::GetModuleProfiler(const std::string& module_name) const {
  auto it = module_profilers_.find(module_name);
  if (it != module_profilers_.end()) {
    return it->second.get();
  }
  return nullptr;
}

/**
 * 在 BuildPipeline 后调用
 */
bool Pipeline::Start() {
  if (IsRunning()) {
    LOGW(CORE) << "Pipeline is running, the Pipeline::Start function is called multiple times.";
    return false;
  }

  // open modules
  bool open_module_failed = false;
  std::vector<std::shared_ptr<Module>> opened_modules;
  for (auto node = graph_->DFSBegin(); node != graph_->DFSEnd(); ++node) {
    if (!node->data.module->Open(node->GetConfig().parameters)) {
      LOGE(CORE) << node->data.module->GetName() << " open failed!";
      open_module_failed = true;
      break;
    }
    opened_modules.push_back(node->data.module);
  }
  if (open_module_failed) {
    for (auto it : opened_modules) it->Close();
    return false;
  }

  running_.store(true);  // 控制位，发出运行指令
  event_bus_->Start();

  // start data transmit
  for (auto node = graph_->DFSBegin(); node != graph_->DFSEnd(); ++node) {
    if (!node->data.parent_nodes_mask) continue;  // head node
    node->data.module->GetConnector()->Start();
  }

  // create process threads
  for (auto node = graph_->DFSBegin(); node != graph_->DFSEnd(); ++node) {
    if (!node->data.parent_nodes_mask) continue;  // head node
    const auto& config = node->GetConfig();
    for (int conveyor_idx = 0; conveyor_idx < config.parallelism; ++conveyor_idx) {
      threads_.push_back(std::thread(&Pipeline::TaskLoop, this, &node->data, conveyor_idx));
    }
  }
  LOGI(CORE) << "Pipeline[" << GetName() << "] " << "Start";
  return true;
}

bool Pipeline::Stop() {
  LOGI(CORE) << "Pipeline[" << GetName() << "] " << "Ready to stop";
  if (!IsRunning()) return true;

  // frist close head module
  for (auto node = graph_->DFSBegin(); node != graph_->DFSEnd(); ++node) {
    if (!node->data.parent_nodes_mask) {
      node->data.module->Close();
    }
  }
  // stop task loop thread
  running_.store(false);

  // stop data transmit
  for (auto node = graph_->DFSBegin(); node != graph_->DFSEnd(); ++node) {
    if (!node->data.parent_nodes_mask) continue;  // head node
    auto connector = node->data.module->GetConnector();
    if (connector) {
      // push data will be rejected after Stop()
      // stop first to ensure connector will be empty
      connector->Stop();
      connector->EmptyDataQueue();
    }
  }

  for (std::thread& it : threads_) {
    if (it.joinable()) it.join();
  }
  threads_.clear();
  // event_bus_->Stop();  // Sasha: 此处暂不调用

  // close other modules
  for (auto node = graph_->DFSBegin(); node != graph_->DFSEnd(); ++node) {
    if (!node->data.parent_nodes_mask) continue;
    node->data.module->Close();
  }

  // clear callback function, important! Especially for the case of using the python api,
  // the callback function will manage the life cycle of a python object.
  // When a circular reference occurs, GC(python) cannot handle it, resulting in a memory leak.
  RegisterFrameDoneCallBack(NULL);
  LOGI(CORE) << "Pipeline[" << GetName() << "] " << "Stop complete";
  return true;
}

Module* Pipeline::GetModule(const std::string& module_name) const {
  auto node = graph_->GetNodeByName(module_name);
  if (node.get()) return node->data.module.get();
  return nullptr;
}

CNModuleConfig Pipeline::GetModuleConfig(const std::string& module_name) const {
  auto node = graph_->GetNodeByName(module_name);
  if (node.get()) return node->GetConfig();
  return {};
}

/**
 * @brief Module 通过 Pipeline 传输数据的入口
 */
bool Pipeline::ProvideData(const Module* module, const std::shared_ptr<FrameInfo> data) {
  // check running.
  if (!IsRunning()) {
    LOGE(CORE) << "[" << module->GetName() << "]" << " Provide data to pipeline [" << GetName() << "] failed, "
        << "pipeline is not running, start pipeline first. " << data->stream_id;
    return false;
  }
  // check module is created by current pipeline.
  if (!module || module->GetContainer() != this) {
    LOGE(CORE) << "Provide data to pipeline [" << GetName() << "] failed, "
        << (module ? ("module named [" + module->GetName() + "] is not created by current pipeline.") :
        "module can not be nullptr.");
    return false;
  }
  // data can only created by root nodes.
  // 非根节点时，module_mask 标记了已经过的节点，在后续 TransmitData - MarkPassed 中会被更新
  // 根节点时，module_mask 标记的所有 module 中不会经过节点
  // 两种节点的设置都会在后续的 TransmitData 中，这里要求只有根节点才能创建 modules_mask_ 为 0 的全新数据
  if (!data->GetModulesMask() && module->context_->parent_nodes_mask) {
    LOGE(CORE) << "Provide data to pipeline [" << GetName() << "] failed, "
        << "Data created by module named [" << module->GetName() << "]. "
        << "Data can be provided to pipeline only when the data is created by root nodes.";
    return false;
  }
  TransmitData(module->context_, data);
  return true;
}

bool Pipeline::IsRootNode(const std::string& module_name) const {
  auto module = GetModule(module_name);
  if (!module) return false;
  return !module->context_->parent_nodes_mask;
}

bool Pipeline::IsLeafNode(const std::string& module_name) const {
  auto module = GetModule(module_name);
  if (!module) return false;
  if (module->context_->node.expired()) {
    LOGE(CORE) << "Module named [" << module_name << "] is not created by current pipeline.";
    return false;
  } else {
    auto p = module->context_->node.lock();
    if (!p) {
      LOGE(CORE) << "Module named [" << module_name << "] is not created by current pipeline.";
      return false;
    }
    return p->GetNext().empty();
  }
  return false;
}

bool Pipeline::CreateModules(std::vector<std::shared_ptr<Module>>* modules) {
  all_modules_mask_ = 0;

  /**
   * 我们通过迭代器访问的都是 std::shared_ptr<typename CNGraph<NodeContext>::CNNode>
   * 遍历过程中会增加 graph_ 中对 CNNode 的引用计数
   */
  for (auto node_iter = graph_->DFSBegin(); node_iter != graph_->DFSEnd(); ++node_iter) {
    const CNModuleConfig& config = node_iter->GetConfig();
    Module* module = ModuleFactory::Instance()->Create(config.className, node_iter->GetFullName());
    if (!module) {
      LOGE(CORE) << "Create module failed, module name : [" << config.name
          << "], class name : [" << config.className << "].";
      return false;
    }
    module->context_ = &node_iter->data;  // NodeContext
    node_iter->data.node = *node_iter;  // 反向引用到 CNNode, 但是不增加引用
    node_iter->data.parent_nodes_mask = 0;
    node_iter->data.route_mask = 0;
    node_iter->data.module = std::shared_ptr<Module>(module);  // 完全控制, Module 是在图中遍历的
    node_iter->data.module->SetContainer(this);
    modules->push_back(node_iter->data.module);
    all_modules_mask_ |= 1UL << node_iter->data.module->GetId();
  }
  return true;
}

std::vector<std::string> Pipeline::GetSortedModuleNames() {
  if (sorted_module_names_.empty()) {
    sorted_module_names_ = graph_->TopoSort();
  }
  return sorted_module_names_;
}

/**
 * 在 BuildPipeline 中调用，用于生成每个模块的 parent_nodes_mask
 */
void Pipeline::GenerateModulesMask() {
  // parent mask helps to determine whether the data has passed all the parent nodes.
  for (auto cur_node = graph_->DFSBegin(); cur_node != graph_->DFSEnd(); ++cur_node) {
    const auto& next_nodes = cur_node->GetNext();
    for (const auto& next : next_nodes) {
      next->data.parent_nodes_mask |= 1UL << cur_node->data.module->GetId();
    }
  }

  // route mask helps to mark that the data has passed through all unreachable nodes.
  // consider the case of multiple head nodes. (multiple source modules)
  for (auto head : graph_->GetHeads()) {
    for (auto iter = head->DFSBegin(); iter != head->DFSEnd(); ++iter) {
      head->data.route_mask |= 1UL << iter->data.module->GetId();
    }
  }
  // 在 head nodes 中标记可达的节点
}

bool Pipeline::CreateConnectors() {
  // node_iter: std::shared_ptr<CNNode> 
  for (auto node_iter = graph_->DFSBegin(); node_iter != graph_->DFSEnd(); ++node_iter) {
    if (!node_iter->data.parent_nodes_mask) continue;  // head nodes
    const auto &config = node_iter->GetConfig();
    // check if parallelism and max_input_queue_size is valid.
    if (config.parallelism <= 0 || config.maxInputQueueSize <= 0) {
      LOGE(CORE) << "Module [" << config.name << "]: parallelism or max_input_queue_size is not valid, "
                  "parallelism[" << config.parallelism << "], "
                  "max_input_queue_size[" << config.maxInputQueueSize << "].";
      return false;
    }
    node_iter->data.module->SetConnector(std::make_shared<Connector>(config.parallelism, config.maxInputQueueSize));
  }
  return true;
}

/**
 * 是否经过所有的父节点
 */
static inline
bool PassedByAllParentNodes(NodeContext* context, uint64_t data_mask) {
  uint64_t parent_masks = context->parent_nodes_mask;
  return (data_mask & parent_masks) == parent_masks;
}

void Pipeline::OnProcessStart(NodeContext* context, const std::shared_ptr<FrameInfo>& data) {
  if (data->IsEos()) return;
  if (IsProfilingEnabled()) {
    auto record_key = std::make_pair(data->stream_id, data->timestamp);
    auto profiler = context->module->GetProfiler();
    if (profiler && context->parent_nodes_mask) {  // not head nodes
      profiler->RecordProcessEnd(kINPUT_PROFILER_NAME, record_key);
      profiler->RecordProcessStart(kPROCESS_PROFILER_NAME, record_key);
    }
  }
}

void Pipeline::OnProcessEnd(NodeContext* context, const std::shared_ptr<FrameInfo>& data) {
  if (data->IsEos()) return;
  if (IsProfilingEnabled()) {
    auto profiler = context->module->GetProfiler();
    if (profiler && context->parent_nodes_mask) {
      profiler->RecordProcessEnd(
        kPROCESS_PROFILER_NAME, std::make_pair(data->stream_id, data->timestamp));
    }
  }
  context->module->NotifyObserver(data);
}

void Pipeline::OnProcessFailed(NodeContext* context, const std::shared_ptr<FrameInfo>& data, int ret) {
  auto module_name = context->module->GetName();
  Event e;
  e.type = EventType::EVENT_ERROR;
  e.module_name = module_name;
  e.message = module_name + " process failed, return number: " + std::to_string(ret);
  e.stream_id = data->stream_id;
  e.thread_id = std::this_thread::get_id();
  event_bus_->PostEvent(e);
}

void Pipeline::OnDataInvalid(NodeContext* context, const std::shared_ptr<FrameInfo>& data) {
  auto module = context->module;
  LOGW(CORE) << "[" << GetName() << "]" << " got frame error from " << module->GetName() <<
    " stream_id: " << data->stream_id << ", pts: " << data->timestamp;

  Event e;
  e.type = EventType::EVENT_FRAME_ERROR;
  e.module_name = module->GetName();
  e.message = module->GetName() + " frame failed";
  e.stream_id = data->stream_id;
  e.thread_id = std::this_thread::get_id();
  event_bus_->PostEvent(e);
  
  // StreamMsg msg;
  // msg.type = StreamMsgType::FRAME_ERR_MSG;
  // msg.stream_id = data->stream_id;
  // msg.module_name = module->GetName();
  // msg.pts = data->timestamp;
  // UpdateByStreamMsg(msg);
}

void Pipeline::OnEos(NodeContext* context, const std::shared_ptr<FrameInfo>& data) {
  auto module = context->module;
  module->NotifyObserver(data);

  if (IsProfilingEnabled())
    module->GetProfiler()->OnStreamEos(data->stream_id);

  LOGI(CORE) << "[" << module->GetName() << "]" << " [" << data->stream_id << "] got eos.";
  // eos message
  Event e;
  e.type = EventType::EVENT_EOS;
  e.module_name = module->GetName();
  e.stream_id = data->stream_id;
  e.thread_id = std::this_thread::get_id();
  event_bus_->PostEvent(e);
}

/**
 * 仅在 Pipeline::TransmitData 中调用
 */
void Pipeline::OnPassThrough(NodeContext* context, const std::shared_ptr<FrameInfo>& data) {
  if (frame_done_cb_) frame_done_cb_(data);  // To notify the frame is processed by all modules
  if (data->IsEos()) {
    // OnEos(context, data);
  }
#ifdef UNIT_TEST
  LOGD(CORE) << "[" << context->module->GetName() << "]" << " [" << data->stream_id << "] pass through all modules; data->IsEos = " << std::boolalpha << data->IsEos();
#endif
}

/**
 * @note: 数据传输的核心函数，在 Module 处理完后
 * 仅在 Pipeline::ProvideData 中调用
 */
void Pipeline::TransmitData(NodeContext* context, const std::shared_ptr<FrameInfo>& data) {
  if (data->IsInvalid()) {
    OnDataInvalid(context, data);
    return;
  }
  if (!context->parent_nodes_mask) {
    // head node
    // set mask to 1 for never touched modules, for case which has multiple source modules.
    data->SetModulesMask(all_modules_mask_ ^ context->route_mask);
  }
  // 如果数据经过的是头结点，那么 
  // SetModulesMask 标记当前根节点不会经过的节点
  // 异或：相同为 0，不同为 1

  if (data->IsEos()) {
    OnEos(context, data);
  } else {
    if (IsStreamRemoved(data->stream_id))  // 表示当前 stream_id 是需要移除的
      return;
  }
  OnProcessEnd(context, data);

  // auto node = context->node.lock();
  if (context->node.expired()) {
    LOGE(CORE) << "NodeContext[" << context->module->GetName() << "] is expired.";
    return;
  }
  auto node = context->node.lock();
  if (!node) {
    LOGE(CORE) << "NodeContext[" << context->module->GetName() << "] is expired.";
    return;
  }
  auto module = context->module;
  const uint64_t cur_mask = data->MarkPassed(module.get());
  const bool passed_by_all_modules = PassedByAllModules(cur_mask);

  if (passed_by_all_modules) {
    OnPassThrough(context, data);  // 不需要再调用 OnEos 操作
    return;
  }

  // transmit to next nodes
  // 操作后面的
  for (auto next_node : node->GetNext()) {
    if (!PassedByAllParentNodes(&next_node->data, cur_mask)) continue;
    auto next_module = next_node->data.module;
    auto connector = next_module->GetConnector();
    // push data to conveyor only after data passed by all parent nodes.

    if (IsProfilingEnabled() && !data->IsEos()) {
      next_module->GetProfiler()->RecordProcessStart(
        kINPUT_PROFILER_NAME, std::make_pair(data->stream_id, data->timestamp));
    }

    const int conveyor_idx = data->GetStreamIndex() % connector->GetConveyorCount();
    while (connector->IsRunning() && connector->PushDataBufferToConveyor(conveyor_idx, data) == false) {
      if (connector->GetFailTime(conveyor_idx) % 10 == 0) {
        // Show infomation when conveyor is full in every second
        LOGD(CORE) << "[" << next_module->GetName() << " " << conveyor_idx << "] " << "Input buffer is full";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }  // while try push
  }  // loop next nodes
}

void Pipeline::TaskLoop(NodeContext* context, uint32_t conveyor_idx) {
  auto module = context->module;
  auto connector = module->GetConnector();
  auto node_name = module->GetName();

  // process loop
  while (IsRunning()) {
    std::shared_ptr<FrameInfo> data = nullptr;
    while (connector->IsRunning() && data == nullptr) {
      data = connector->PopDataBufferFromConveyor(conveyor_idx);
    }
    if (!connector->IsRunning())
      break;
    if (data == nullptr)
      continue;
    OnProcessStart(context, data);
    int ret = module->DoProcess(data);
    if (ret < 0)
      OnProcessFailed(context, data, ret);
  }  // while process loop
}

EventHandleFlag Pipeline::DefaultBusWatch(const Event& event) {
  StreamMsg smsg;
  EventHandleFlag ret;
  switch (event.type) {
    case EventType::EVENT_ERROR:
      smsg.type = StreamMsgType::ERROR_MSG;
      smsg.module_name = event.module_name;
      smsg.stream_id = event.stream_id;
      UpdateByStreamMsg(smsg);
      LOGE(CORE) << "[" << event.module_name << "]: "
                 << event.message;
      ret = EventHandleFlag::EVENT_HANDLE_STOP;
      break;
    case EventType::EVENT_STOP:
      LOGI(CORE) << "[" << event.module_name << "]: "
                 << event.message;
      ret = EventHandleFlag::EVENT_HANDLE_STOP;
      break;
    // EVENT_ERROR 和 EVENT_STOP 导致 EventBus 停止
    case EventType::EVENT_WARNING:
      LOGW(CORE) << "[" << event.module_name << "] " << event.message;
      ret = EventHandleFlag::EVENT_HANDLE_SYNCED;
      break;
    case EventType::EVENT_EOS: {
      LOGD(CORE) << "Pipeline received eos from module " + event.module_name << " of stream " << event.stream_id;
      smsg.type = StreamMsgType::EOS_MSG;
      smsg.module_name = event.module_name;
      smsg.stream_id = event.stream_id;
      UpdateByStreamMsg(smsg);  // 执行 EOS 逻辑
      ret = EventHandleFlag::EVENT_HANDLE_SYNCED;
      break;
    }
    case EventType::EVENT_STREAM_ERROR: {
      smsg.type = StreamMsgType::STREAM_ERR_MSG;
      smsg.module_name = event.module_name;
      smsg.stream_id = event.stream_id;
      UpdateByStreamMsg(smsg);
      LOGD(CORE) << "Pipeline received stream error from module " + event.module_name
                 << " of stream " << event.stream_id;
      ret = EventHandleFlag::EVENT_HANDLE_SYNCED;
      break;
    }
    case EventType::EVENT_INVALID:
      LOGE(CORE) << "[" << event.module_name << "]: "
                 << event.message;
    default:
      ret = EventHandleFlag::EVENT_HANDLE_NULL;
      break;
  }
  return ret;
}

void Pipeline::UpdateByStreamMsg(const StreamMsg& msg) {
  LOGD(CORE) << "[" << GetName() << "] "
             << "stream: " << msg.stream_id << " got message: " << static_cast<std::size_t>(msg.type);
  msgq_.Push(msg);
}

/**
 * Pipeline 级别的消息处理执行，用于操作 Pipeline 内的成员，例如停止流
 * EventBus 用于产生执行消息
 */
void Pipeline::StreamMsgHandleFunc() {
  while (!exit_msg_loop_) {
    StreamMsg msg;
    while (!exit_msg_loop_ && !msgq_.WaitAndTryPop(msg, std::chrono::milliseconds(200))) {
    }

    if (exit_msg_loop_) {
        LOGI(CORE) << "[" << GetName() << "] stop updating stream message";
        return;
    }
    switch (msg.type) {
      case StreamMsgType::EOS_MSG:
      case StreamMsgType::ERROR_MSG:
      case StreamMsgType::STREAM_ERR_MSG:
      case StreamMsgType::FRAME_ERR_MSG:
      case StreamMsgType::USER_MSG0:
      case StreamMsgType::USER_MSG1:
      case StreamMsgType::USER_MSG2:
      case StreamMsgType::USER_MSG3:
      case StreamMsgType::USER_MSG4:
      case StreamMsgType::USER_MSG5:
      case StreamMsgType::USER_MSG6:
      case StreamMsgType::USER_MSG7:
      case StreamMsgType::USER_MSG8:
      case StreamMsgType::USER_MSG9:
        LOGD(CORE) << "[" << GetName() << "]" << " stream: " << msg.stream_id 
                   << " notify message: " << static_cast<std::size_t>(msg.type);
        if (smsg_observer_) {
          smsg_observer_->Update(msg);
        }
        break;
      default:
        break;
    }
  }
}

}  // namespace cnstream