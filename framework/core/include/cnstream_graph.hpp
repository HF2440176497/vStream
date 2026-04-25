/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
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

#ifndef CNSTREAM_GRAPH_HPP_
#define CNSTREAM_GRAPH_HPP_

/**
 * @file cnstream_graph.hpp
 *
 * This file contains a declaration of the CNGraph class.
 */

#include <climits>
#include <cstdlib>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_config.hpp"
#include "cnstream_logging.hpp"

namespace cnstream {

/**
 * @class DAGAlgorithm
 *
 * @brief DAGAlgorithm is class representing the DAG algorithm implementation.
 */
class DAGAlgorithm {
 public:
  /**
   * @class DFSIterator
   *
   * @brief DFSIterator is a class of iterator for traversing a DAG with DFS order.
   */
  class DFSIterator {
   public:
    /**
     * @brief Steps into the next iterator in DFS order.
     *
     * @return Returns the reference of the current iterator.
     */
    DFSIterator& operator++();
    /**
     * @brief Jedges if the current iterator is equal to the other one.
     *
     * @param[in] other The iterator to be compared.
     *
     * @return Returns true if the current iterator is equal to `other`. Otherwise, returns false.
     */
    bool operator==(const DFSIterator& other) const;
    /**
     * @brief Jedges if the current iterator is not equal to the other one.
     *
     * @param[in] other The iterator to be compared.
     *
     * @return Returns true if the current iterator is not equal to `other`. Otherwise, returns false.
     */
    bool operator!=(const DFSIterator& other) const;
    /**
     * @brief Returns the vertex that the current iterator points to.
     *
     * @return Returns the vertex that the current iterator points to.
     */
    int operator*() const;

   private:
    friend class DAGAlgorithm;
    explicit DFSIterator(const DAGAlgorithm* dag) : dag_(dag) {}
    std::stack<int> vertex_stack_;
    std::vector<bool> visit_;
    const DAGAlgorithm* dag_ = nullptr;
  };  // class DFSIterator
  /**
   * @brief Reserves vertex memory.
   *
   * @param[in] num_vertices The number of vertices.
   */
  void Reserve(size_t num_vertices);
  /**
   * @brief Add vertex to DAG.
   *
   * @return Returns the vertex index that incremented from 0.
   */
  int AddVertex();
  /**
   * @brief Add edge to DAG.
   *
   * @param[in] vertexa The start vertex index of edge.
   * @param[in] vertexb The end vertex index of edge.
   *
   * @return Returns true for success. Returns false when the specified endpoint does not exist.
   */
  bool AddEdge(int vertexa, int vertexb);
  /**
   * @brief Gets the indegree of the specified vertex.
   *
   * @param[in] vertex The vertex index.
   *
   * @return Returns the indegree of the specified vertex. -1 will be returned when the vertex not exist.
   */
  int GetIndegree(int vertex) const;
  /**
   * @brief Gets the outdegree of the specified vertex.
   *
   * @param[in] vertex The vertex index.
   *
   * @return Returns the outdegree of the specified vertex. -1 will be returned when the vertex does not exist.
   */
  int GetOutdegree(int vertex) const;
  /**
   * @brief Gets the head vertices.
   *
   * @return Returns head vertices.
   */
  std::vector<int> GetHeads() const;
  /**
   * @brief Gets the tail vertices.
   *
   * @return Returns tail vertices.
   */
  std::vector<int> GetTails() const;
  /**
   * @brief Topological sorting.
   *
   * @return Returns the topology sorting results.
   * Return value is a std::pair object, the first value stored sorted vertices
   * and the second value stored unsorted vertices.
   */
  std::pair<std::vector<int>, std::vector<int>> TopoSort() const;
  /**
   * @brief Gets the begin iterator in DFS order.
   *
   * @param None.
   *
   * @return Returns the begin iterator in DFS order.
   */
  DFSIterator DFSBegin() const;
  /**
   * @brief Gets the start iterator with specified vertex in DFS order.
   *
   * @param[in] vertex The vertex index.
   *
   * @return Returns the start iterator with specified vertex in DFS order.
   */
  DFSIterator DFSBeginFrom(int vertex) const;
  /**
   * @brief Gets the end iterator in DFS order.
   *
   * @return Returns the end iterator in DFS order.
   */
  DFSIterator DFSEnd() const;

 private:
  std::vector<std::set<int>> edges_;  // index: vertax_a , std::set 是定点a到的其他点
  std::vector<int> indegrees_;
};  // class DAGAlgorithm

/**
 * @class CNGraph
 *
 * @brief CNGraph is a class for build DAG by a graph configuration.
 */
template<typename T>
class CNGraph {
 public:
  class CNNode;
  /**
   * @class DFSIterator
   *
   * @brief DFSIterator is a class of iterator for traversing a graph with DFS order.
   */
  class DFSIterator {
   public:
    /**
     * @brief Steps into the next iterator in DFS order.
     *
     * @return Returns the reference of the current iterator.
     */
    DFSIterator& operator++();
    /**
     * @brief Jedges if the current iterator is equal to the other one.
     *
     * @param[in] other The iterator to be compared.
     *
     * @return Returns true if the current iterator is equal to `other`. Otherwise, returns false.
     */
    bool operator==(const DFSIterator& other) const;
    /**
     * @brief Jedges if the current iterator is not equal to the other one.
     *
     * @param[in] other The iterator to be compared.
     *
     * @return Returns true if the current iterator is not equal to `other`. Otherwise, returns false.
     */
    bool operator!=(const DFSIterator& other) const;
    /**
     * @brief Returns the shared pointer of node that the current iterator points to.
     *
     * @param None.
     *
     * @return Returns the shared pointer of node that the current iterator points to.
     */
    std::shared_ptr<CNNode> operator*() const;
    /**
     * @brief Returns the pointer of node that the current iterator points to.
     *
     * @param None.
     *
     * @return Returns the pointer of node that the current iterator points to.
     */
    CNNode* operator->() const;

   private:
    friend class CNGraph;
    explicit DFSIterator(const CNGraph* graph);
    /**
     * called by operator++, returns true when next step is a new node or end node.
     **/
    bool DAGStep();
    const CNGraph* graph_ = nullptr;
    DAGAlgorithm::DFSIterator dag_iter_;
  };  // class DFSIterator
  /**
   * @class CNNode
   *
   * @brief CNNode is a class describing graph node.
   */
  class CNNode {
   public:
    T data;  ///< custom data
    /**
     * @brief Gets the name of node without a graph name prefix.
     */
    std::string GetName() const;
    /**
     * @brief Gets the full name of node with a graph name prefix divided by slashs.
     * eg. root_graph_name/node_name.
     */
    std::string GetFullName() const;
    /**
     * @brief Gets the node configuration(module configuration).
     */
    const CNModuleConfig& GetConfig() const;
    /**
     * @brief Gets next nodes.
     */
    const std::set<std::shared_ptr<CNNode>>& GetNext() const;
    /**
     * @brief Gets the iterator begin from current node in DFS order.
     */
    DFSIterator DFSBegin() const;
    /**
     * @brief Gets the end iterator.
     */
    DFSIterator DFSEnd() const;

#ifdef VSTREAM_UNIT_TEST
   public:
#else
   private:
#endif
    friend class CNGraph;
    explicit CNNode(const CNGraph* const graph) : graph_(graph) {}
    /**
     * @brief Gets the root graph which contains current node.
     */
    const CNGraph* GetRootGraph() const;
    CNModuleConfig config_;
    const CNGraph* const graph_;
    std::set<std::shared_ptr<CNNode>> next_;
  };  // class CNNode

  /**
   * @brief Default constructor to construct one graph.
   */
  CNGraph() = default;
  /**
   * @brief Constructs one graph.
   *
   * @param[in] config The graph configuration.
   */
  explicit CNGraph(const CNGraphConfig& config) : config_(config) {}
  /**
   * @brief Constructs one graph.
   *
   * @param[in] config The graph configuration(rvalue reference).
   */
  explicit CNGraph(CNGraphConfig&& config) : config_(std::forward<CNGraphConfig>(config)) {}
  /**
   * @brief Clears current graph.
   */
  void Clear();
  /**
   * @brief Initializes the current graph by a specified configuration.
   *
   * @param[in] config The graph configuration.
   *
   * @return Returns true for success. Otherwise, returns false.
   * See ::CNGraph::Init for the cases of return false.
   */
  bool Init(const CNGraphConfig& config);
  /**
   * @brief Initializes the current graph by a specified configuration.
   *
   * @param[in] config The graph configuration(rvalue reference).
   *
   * @return Returns true for success. Otherwise, returns false.
   * See ::CNGraph::Init for the cases of returning false.
   */
  bool Init(CNGraphConfig&& config);
  /**
   * @brief Initializes the current graph by the graph configuration set at construction time.
   *
   * @return Returns true for success.
   */
  bool Init();
  /**
   * @brief Determines whether it is an empty graph.
   *
   * @return Returns true if the current graph is an empty graph. Otherwise, returns false.
   */
  bool Empty() const;
  /**
   * @brief Gets graph configuration.
   *
   * @return Returns graph configuration.
   */
  const CNGraphConfig& GetConfig() const;
  /**
   * @brief Gets profiler configuration.
   *
   * @return Returns profiler configuration.
   */
  const ProfilerConfig& GetProfilerConfig() const;
  /**
   * @brief Gets graph name.
   *
   * @return Returns graph name.
   */
  std::string GetName() const;
  /**
   * @brief Gets graph name with a parent graph name prefix divided by slashs.
   *
   * @return Returns graph name with a parent graph name prefix divided by slashs.
   * eg. root_graph_name/parent_graph_name/current_graph_name.
   */
  std::string GetFullName() const;
  /**
   * @brief Gets head nodes.
   *
   * @return Returns head nodes.
   */
  const std::vector<std::shared_ptr<CNNode>>& GetHeads() const;
  /**
   * @brief Gets tail nodes.
   *
   * @return Returns tail nodes.
   */
  const std::vector<std::shared_ptr<CNNode>>& GetTails() const;
  /**
   * @brief Gets a node in current graph by name.
   *
   * @param[in] name The name specified in the node configuration.
   * If you specify a node name in the module configuration, the first node with the same name as
   * the specified node name in the order of DFS will be returned.
   * 
   * @return Returns the node if the module named ``name`` has been added to
   *         the current graph. Otherwise, returns nullptr.
   */
  std::shared_ptr<CNNode> GetNodeByName(const std::string& name) const;
  /**
   * @brief Gets the begin iterator in DFS order.
   */
  DFSIterator DFSBegin() const;
  /**
   * @brief Gets the end iterator.
   */
  DFSIterator DFSEnd() const;
  std::vector<std::string> TopoSort() const;

 private:
  DFSIterator DFSBeginFrom(const CNNode* node) const;
  DFSIterator DFSBeginFrom(std::string&& node_name) const;
  std::string GetLogPrefix() const;
  using ModuleNode = std::pair<int, std::shared_ptr<CNNode>>;  // first: vertex id
  bool AddVertex(const CNModuleConfig& config);
  void AddEdge(const ModuleNode& modulea, const ModuleNode& moduleb);
  bool InitEdges();
  void FindHeadsAndTails();

#ifdef VSTREAM_UNIT_TEST
  public:
#else
  private:
#endif
  std::map<std::string, ModuleNode> module_node_map_;
  std::vector<std::string> vertex_map_to_node_name_;
  std::vector<std::shared_ptr<CNNode>> heads_, tails_;
  CNGraphConfig config_;
  DAGAlgorithm dag_algorithm_;
  const CNGraph* parent_graph_ = nullptr;  // 同时充当标志位
};  // class CNGraph

inline
void DAGAlgorithm::Reserve(size_t num_vertices) {
  edges_.reserve(num_vertices);
}

inline
int DAGAlgorithm::AddVertex() {
  edges_.emplace_back(std::set<int>());
  indegrees_.push_back(0);
  return edges_.size() - 1;
}

/**
 * 需要保证 veretx 已经 Add 进去
 */
inline
bool DAGAlgorithm::AddEdge(int vertexa, int vertexb) {
  int num_vertices = edges_.size();
  if (vertexa >= num_vertices || vertexb >= num_vertices) return false;
  if (!edges_[vertexa].insert(vertexb).second) return false;
  indegrees_[vertexb]++;
  return true;
}

inline
int DAGAlgorithm::GetIndegree(int vertex) const {
  return static_cast<int>(indegrees_.size()) > vertex ? indegrees_[vertex] : -1;
}

inline
int DAGAlgorithm::GetOutdegree(int vertex) const {
  return static_cast<int>(edges_.size()) > vertex ? edges_[vertex].size() : -1;
}

inline
DAGAlgorithm::DFSIterator DAGAlgorithm::DFSEnd() const {
  return DFSIterator(this);
}

inline
bool DAGAlgorithm::DFSIterator::operator!=(const DAGAlgorithm::DFSIterator& other) const {
  return !(*this == other);
}

inline
int DAGAlgorithm::DFSIterator::operator*() const {
  return vertex_stack_.empty() ? -1 : vertex_stack_.top();
}

template<typename T> inline
CNGraph<T>::DFSIterator::DFSIterator(const CNGraph<T>* graph) : graph_(graph),
    dag_iter_(graph->dag_algorithm_.DFSEnd()) {  // 初始化默认构造
}

template<typename T>
bool CNGraph<T>::DFSIterator::DAGStep() {
  ++dag_iter_;  // DFS 迭代
  if (graph_->dag_algorithm_.DFSEnd() == dag_iter_) return true;
  auto node_name = graph_->vertex_map_to_node_name_[*dag_iter_];
  return true;
}

template<typename T>
typename CNGraph<T>::DFSIterator&
CNGraph<T>::DFSIterator::operator++() {
  while (true) {
    if (graph_->dag_algorithm_.DFSEnd() == dag_iter_)  break;
      // current node is a module
    if (DAGStep()) break;
  }
  return *this;
}

template<typename T> inline
bool CNGraph<T>::DFSIterator::operator==(const CNGraph<T>::DFSIterator& other) const {
  return graph_ == other.graph_ && dag_iter_ == other.dag_iter_;
}

template<typename T> inline
bool CNGraph<T>::DFSIterator::operator!=(const typename CNGraph<T>::DFSIterator& other) const {
  return !(*this == other);
}

/**
 * @brief 根据是否是子图进行解引用，非子图时返回对应 Node
 */
template<typename T> inline
std::shared_ptr<typename CNGraph<T>::CNNode>
CNGraph<T>::DFSIterator::operator*() const {
  if (graph_->DFSEnd() == *this) return nullptr;
  auto vertex = *dag_iter_;
  return std::const_pointer_cast<typename CNGraph<T>::CNNode>(
    graph_->module_node_map_.find(graph_->vertex_map_to_node_name_[vertex])->second.second);
  return nullptr;
}

template<typename T> inline
typename CNGraph<T>::CNNode*
CNGraph<T>::DFSIterator::operator->() const {
  return this->operator*().get();
}

template<typename T> inline
std::string CNGraph<T>::CNNode::GetName() const {
  return config_.name;
}

template<typename T> inline
std::string CNGraph<T>::CNNode::GetFullName() const {
  return graph_->GetFullName() + "/" + GetName();
}

template<typename T> inline
const CNModuleConfig& CNGraph<T>::CNNode::GetConfig() const {
  return config_;
}

template<typename T> inline
const std::set<std::shared_ptr<typename CNGraph<T>::CNNode>>&
CNGraph<T>::CNNode::GetNext() const {
  return next_;
}

template<typename T> inline
typename CNGraph<T>::DFSIterator CNGraph<T>::CNNode::DFSBegin() const {
  return GetRootGraph()->DFSBeginFrom(this);
}

template<typename T> inline
typename CNGraph<T>::DFSIterator CNGraph<T>::CNNode::DFSEnd() const {
  return GetRootGraph()->DFSEnd();
}

template<typename T>
const CNGraph<T>* CNGraph<T>::CNNode::GetRootGraph() const {
  const CNGraph* root_graph = graph_;
  while (root_graph->parent_graph_) root_graph = root_graph->parent_graph_;
  return root_graph;
}

namespace __help_functions__ {

/**
 * @brief 检查节点名称是否合法
 * @details 应当不包含 '/' 或 ':'
 */
inline bool IsNodeNameValid(const std::string& name) {
  std::string t;
  t = name;
  return t.find('/') == std::string::npos && t.find(':') == std::string::npos;
}

inline std::string GetRealPath(const std::string& path) {
  char out[PATH_MAX] = {'\0'};
  char* ret = realpath(path.c_str(), out);
  if (NULL == ret) {
    LOGE(CORE) << "Get real path failed, error msg: " << strerror(errno)
               << ". Origin path str: " << path;
    return "";
  }
  return std::string(out);
}

}  // namespace __help_functions__

template<typename T> inline
void CNGraph<T>::Clear() {
  module_node_map_.clear();
  vertex_map_to_node_name_.clear();
  heads_.clear();
  tails_.clear();
  dag_algorithm_ = DAGAlgorithm();
}

template<typename T> inline
bool CNGraph<T>::Init(const CNGraphConfig& config) {
  config_ = config;
  return Init();
}

template<typename T> inline
bool CNGraph<T>::Init(CNGraphConfig&& config) {
  config_ = std::forward<CNGraphConfig>(config);
  return Init();
}

template<typename T>
bool CNGraph<T>::Init() {
  Clear();
  dag_algorithm_.Reserve(config_.module_configs.size());
  // insert vertices
  for (const auto& module_config : config_.module_configs) {
    if (!AddVertex(module_config)) return false;
  }

  if (!InitEdges()) return false;

  FindHeadsAndTails();

  // check circle
  auto topo_result = dag_algorithm_.TopoSort();
  if (topo_result.second.size()) {
    LOGE(CORE) << GetLogPrefix() + "Ring detected.";
    return false;
  }
  return true;
}

template<typename T> inline
bool CNGraph<T>::Empty() const {
  return vertex_map_to_node_name_.empty();
}

template<typename T> inline
const CNGraphConfig& CNGraph<T>::GetConfig() const {
  return config_;
}

template<typename T> inline
const ProfilerConfig& CNGraph<T>::GetProfilerConfig() const {
  return config_.profiler_config;
}

template<typename T> inline
std::string CNGraph<T>::GetName() const {
  return config_.name;
}

template<typename T> inline
std::string CNGraph<T>::GetFullName() const {
  std::string prefix = parent_graph_ ? parent_graph_->GetFullName() : "";
  return prefix.empty() ? GetName() : prefix + "/" + GetName();
}

template<typename T> inline
const std::vector<std::shared_ptr<typename CNGraph<T>::CNNode>>&
CNGraph<T>::GetHeads() const {
  return heads_;
}

template<typename T> inline
const std::vector<std::shared_ptr<typename CNGraph<T>::CNNode>>&
CNGraph<T>::GetTails() const {
  return tails_;
}

template<typename T> inline
std::shared_ptr<typename CNGraph<T>::CNNode>
CNGraph<T>::GetNodeByName(const std::string& name) const {
  std::vector<std::string> v;
  std::string t = "";
  for (const char& c : name) {
    if ('/' == c) {
      v.emplace_back(t);
      t = "";
    } else {
      t.push_back(c);
    }
  }
  v.emplace_back(std::move(t));
  const CNGraph<T>* graph = this;
  if (v.size() == 1) {
    // no graph prefix, search all nodes
    for (DFSIterator it = DFSBegin(); it != DFSEnd(); ++it) {
      if (it->GetName() == v[0]) return *it;
    }
  } else if (v.size() == 2) {
    // has graph prefix
    if (v[0] != GetName()) {
      LOGE(CORE) << "Node named [" + name + "] is not belongs to graph named [" + GetName() + "].";
      return nullptr;
    }
    auto node_iter = graph->module_node_map_.find(v.back());
    if (graph->module_node_map_.end() == node_iter) {
      LOGE(CORE) << "Can not find node named [" << name << "].";
      return nullptr;
    }
    return node_iter->second.second;
  } else if (v.size() > 2) {
    LOGE(CORE) << "Node named [" + name + "] not support subgraph.";
    return nullptr;
  }
  return nullptr;
}

template<typename T>
typename CNGraph<T>::DFSIterator CNGraph<T>::DFSBegin() const {
  DFSIterator iter(this);
  iter.dag_iter_ = dag_algorithm_.DFSBegin();
  while (DFSEnd() != iter) {
    auto node_name = vertex_map_to_node_name_[*iter.dag_iter_];  // 取出 vertex_stack_ 栈顶元素
    break;
  }
  return iter;
}

template<typename T> inline
typename CNGraph<T>::DFSIterator CNGraph<T>::DFSEnd() const {
  DFSIterator iter(this);
  iter.dag_iter_ = dag_algorithm_.DFSEnd();
  return iter;
}

template<typename T>
std::vector<std::string> CNGraph<T>::TopoSort() const {
  std::vector<std::string> results;
  auto sorted_idx = dag_algorithm_.TopoSort().first;  // std::vector<int>
  results.reserve(sorted_idx.size());
  for (auto idx : sorted_idx) {
    std::string node_name = vertex_map_to_node_name_[idx]; 
    results.emplace_back(GetFullName() + "/" + node_name);  
  }
  return results;
}

template<typename T> inline
typename CNGraph<T>::DFSIterator CNGraph<T>::DFSBeginFrom(const CNNode* node) const {
  // be carefully, make sure current graph is the root graph of this node when calling this function.
  return DFSBeginFrom(node->GetFullName());
}

// 涉及到子图的，需要递归调用 DFSBeginFrom 进行
template<typename T>
typename CNGraph<T>::DFSIterator CNGraph<T>::DFSBeginFrom(std::string&& node_full_name) const {
  // no need to check if the node exists in current graph, because the node is created by the graph itself.

  // remove current graph name prefix and the first slash.
  node_full_name = node_full_name.substr(GetName().size() + 1, std::string::npos);
  DFSIterator iter(this);
  size_t slash_pos = node_full_name.find("/");
  iter.dag_iter_ = dag_algorithm_.DFSBeginFrom(module_node_map_.find(node_full_name)->second.first);
  return iter;
}

template<typename T> inline
std::string CNGraph<T>::GetLogPrefix() const {
  return "[Graph:" + GetFullName() + "]: ";
}

/**
 * 在 Init 中调用
 */
template<typename T>
bool CNGraph<T>::AddVertex(const CNModuleConfig& config) {
  if (!__help_functions__::IsNodeNameValid(config.name)) {
    LOGE(CORE) << GetLogPrefix() + "Module[" + config.name + "] name invalid. "
        "The name of modules can not contain slashes or risks";
    return false;
  }
  auto node = std::shared_ptr<CNNode>(new CNNode(this));
  node->config_ = config;

  int vertex_id = dag_algorithm_.AddVertex();  // index_id 可以认为是递增的
  vertex_map_to_node_name_.push_back(config.name);

  if (!module_node_map_.insert(std::make_pair(config.name, std::make_pair(vertex_id, node))).second) {
    LOGE(CORE) << GetLogPrefix() + "Module[" + config.name + "] name duplicated.";
    return false;
  }
  return true;
}

template<typename T> inline
void CNGraph<T>::AddEdge(const ModuleNode& modulea,
    const ModuleNode& moduleb) {
  modulea.second->next_.insert(moduleb.second);
  dag_algorithm_.AddEdge(modulea.first, moduleb.first);
}

/**
 * @brief 按照 Node 类别调用 AddEdge 初始化
 * 调用处：Init
 */
template<typename T>
bool CNGraph<T>::InitEdges() {
  // edges head is a module
  for (const auto& node_pair : module_node_map_) {
    auto cur_node = node_pair.second;
    const std::set<std::string>& next = cur_node.second->config_.next;
    for (const auto& next_node_name : next) {
      // module->module
      auto next_module_iter = module_node_map_.find(next_node_name);
      if (next_module_iter != module_node_map_.end()) {
        AddEdge(cur_node, next_module_iter->second);
      } else {
        LOGE(CORE) << GetLogPrefix() + "Unable to find a downstream node named "
            "[" + next_node_name + "] for module [" + node_pair.first +"].";
        return false;
      }
    }  // for next nodes
  }  // for module_node_map_
  return true;
}

template<typename T>
void CNGraph<T>::FindHeadsAndTails() {
  auto head_vertices = dag_algorithm_.GetHeads();
  for (auto vertex : head_vertices) {
    auto node_name = vertex_map_to_node_name_[vertex];
    heads_.push_back(module_node_map_.find(node_name)->second.second);
  }

  auto tail_vertices = dag_algorithm_.GetTails();
  for (auto vertex : tail_vertices) {
    auto node_name = vertex_map_to_node_name_[vertex];
    tails_.push_back(module_node_map_.find(node_name)->second.second);
  }
}

}  // namespace cnstream

#endif  // CNSTREAM_GRAPH_HPP_

