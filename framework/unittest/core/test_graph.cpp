
/**
 * 图算法单元测试
 */

#include "base.hpp"
#include "cnstream_config.hpp"
#include "cnstream_graph.hpp"
#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"

static std::string test_pipeline_json = "pipeline_config.json";

namespace cnstream {

TEST(CoreDAGAlgorithm, AddVertex) {
  DAGAlgorithm dag;
  dag.Reserve(3);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(i, dag.AddVertex());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(dag.GetIndegree(i), 0);
    EXPECT_EQ(dag.GetOutdegree(i), 0);
  }
}

TEST(CoreDAGAlgorithm, AddEdge) {
  DAGAlgorithm dag;
  for (int i = 0; i < 3; ++i)
    dag.AddVertex();
  // case1: add success
  EXPECT_TRUE(dag.AddEdge(1, 2));
  // case2: vertex out of range 
  EXPECT_FALSE(dag.AddEdge(0, 3));
  EXPECT_FALSE(dag.AddEdge(5, 1));
  // case3: add the same edge twice
  EXPECT_FALSE(dag.AddEdge(1, 2));
}

TEST(CoreDAGAlgorithm, DFSEnd) {
  DAGAlgorithm dag;
  for (int i = 0; i < 3; ++i)
    dag.AddVertex();
  EXPECT_EQ(-1, *dag.DFSEnd());
}


TEST(CoreCNGraph, InitNormalSimpleGraph) {
  // no ring
  /**
   * two source
   *       0   7
   *      / \ /
   *     1   2
   *    /   / \
   *   3   4   5
   *    \     /
   *     \   /
   *       6
   **/
  CNGraphConfig graph_config;
  graph_config.name = "test_graph";
  CNModuleConfig config0;
  config0.name = "0";
  config0.next = {"1", "2"};
  graph_config.module_configs.push_back(config0);
  CNModuleConfig config1;
  config1.name = "1";
  config1.next = {"3"};
  graph_config.module_configs.push_back(config1);
  CNModuleConfig config2;
  config2.name = "2";
  config2.next = {"4", "5"};
  graph_config.module_configs.push_back(config2);
  CNModuleConfig config3;
  config3.name = "3";
  config3.next = {"6"};
  graph_config.module_configs.push_back(config3);
  CNModuleConfig config4;
  config4.name = "4";
  config4.next = {};
  graph_config.module_configs.push_back(config4);
  CNModuleConfig config5;
  config5.name = "5";
  config5.next = {"6"};
  graph_config.module_configs.push_back(config5);
  CNModuleConfig config6;
  config6.name = "6";
  config6.next = {};
  graph_config.module_configs.push_back(config6);
  CNModuleConfig config7;
  config7.name = "7";
  config7.next = {"2"};
  graph_config.module_configs.push_back(config7);
  CNGraph<int> graph(graph_config);
  EXPECT_TRUE(graph.Init());

  std::vector<std::string> expected_heads {"0", "7"};
  EXPECT_EQ(expected_heads.size(), graph.GetHeads().size());
  for (int i = 0; i < expected_heads.size(); ++i)
    EXPECT_EQ(graph.GetHeads()[i]->GetName(), expected_heads[i]);  // Node->GetName()

  /**
   * 重要：AddVertex(module_config) 最终是根据 GraphConfig 根据文件配置顺序
   * Init() 也已经建立好了 edge 关系
   */
  std::vector<std::string> expected_dfs_order = {"7", "2", "4", "5", "6", "0", "1", "3"};
  auto iter = graph.DFSBegin();
  for (size_t i = 0; i < expected_dfs_order.size(); ++i, ++iter) {
    EXPECT_NE(iter, graph.DFSEnd());
    EXPECT_EQ(iter->GetName(), expected_dfs_order[i]);
    std::cout << "[" << i << "] = " << iter->GetName() << " ";
  }
  std::cout << std::endl;
}

class ConfigFileLoad : public testing::Test {
  protected:
    virtual void SetUp() {
      std::string json_content = readFile(test_pipeline_json.c_str());
      EXPECT_FALSE(json_content.empty()) << "Read json file failed";
      cnstream::CNGraphConfig graph_config;
      graph_config.ParseByJSONStr(json_content);
      graph_config_ = graph_config;
    }

    virtual void TearDown() {  // 当前用例结束
      LOGI(ConfigFileLoad) << "TearDown";
    }

  protected:
    cnstream::CNGraphConfig graph_config_;
};

/**
 * 读取配置文件，初始化图，检查图结构
 */
TEST_F(ConfigFileLoad, BaseInitGraph) {
  std::unique_ptr<CNGraph<NodeContext>> graph = std::make_unique<CNGraph<NodeContext>>();
  EXPECT_NE(nullptr, graph.get());
  EXPECT_TRUE(graph->Init(graph_config_));

  // Graph 只包含 Modudle SubGraphNode
  std::vector<std::string> expected_heads {"decoder"};
  std::vector<std::string> expected_nodes {"decoder", "InferenceYolo", "InferenceClass", "sort_h", "osd"};

  // Init 过程会依赖 module_configs 因此我们首先检查
  // 应当和 pipeline.json 的配置对应
  EXPECT_EQ(graph_config_.module_configs.size(), expected_nodes.size());
  for (int i = 0; i < expected_nodes.size(); ++i) {
    std::cout << "[" << i << "] = " << graph_config_.module_configs[i].name << " ";
  }
  std::cout << std::endl;

  EXPECT_EQ(graph->vertex_map_to_node_name_.size(), expected_nodes.size());
  for (int i = 0; i < expected_nodes.size(); ++i) {
    EXPECT_EQ(graph->vertex_map_to_node_name_[i], expected_nodes[i]);
  }

  // Node->GetName() return Node->config_.name
  auto heads = graph->GetHeads();
  EXPECT_EQ(heads.size(), expected_heads.size());
  for (int i = 0; i < heads.size(); ++i) {
    EXPECT_EQ(heads[i]->GetName(), expected_heads[i]);
  }
  
  auto iter = graph->DFSBegin();
  for (int i = 0; iter != graph->DFSEnd(), i < expected_nodes.size(); ++iter, ++i) {
    const CNModuleConfig& config = iter->GetConfig();
    std::cout << "[" << i << "] = " << config.name << " ";
    
    EXPECT_EQ(config.name, iter->GetName());
    EXPECT_EQ(expected_nodes[i], iter->GetName());
    EXPECT_EQ(config.next.size(), iter->next_.size());
  }
  std::cout << std::endl;
}

}   // end namespace cnstream
