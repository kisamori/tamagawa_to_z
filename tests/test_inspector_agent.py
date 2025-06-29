"""Inspector-Validator Agent のテストモジュール

このモジュールは、Inspector-Validator Agentの各コンポーネントの
動作を検証するテストを提供します。
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# テスト対象のインポート
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tamagawa_to_z.inspector_agent.metrics import (
    recall_at_k, map_score, workload, root_diversity,
    calculate_all_metrics, analyze_spatial_distribution
)
from tamagawa_to_z.inspector_agent.agent_schema import (
    validate_action_params, PRESET_ACTIONS
)
from tamagawa_to_z.inspector_agent.agent import InspectorValidatorAgent


class TestMetrics(unittest.TestCase):
    """メトリクス計算のテストクラス"""
    
    def setUp(self):
        """テスト用データの準備"""
        # 候補データ（CSV形式）
        self.candidates_data = pd.DataFrame({
            'name': ['Rio Test', 'Lagoa Sample', 'Igarape Demo'],
            'geometry': [
                'POINT (-67.0 -10.0)',
                'POINT (-67.1 -10.1)', 
                'POINT (-67.2 -10.2)'
            ],
            'total_score': [0.9, 0.7, 0.5],
            'root': ['rio', 'lagoa', 'igarape']
        })
        
        # 既知遺跡データ（GeoDataFrame）
        self.known_sites = gpd.GeoDataFrame({
            'site_id': [1, 2, 3],
            'geometry': [
                Point(-67.001, -10.001),  # Rio Testに近い
                Point(-67.5, -10.5),      # どの候補からも遠い
                Point(-67.201, -10.201)   # Igarape Demoに近い
            ]
        }, crs="EPSG:4326")
    
    def test_recall_at_k(self):
        """Recall@K指標のテスト"""
        # K=2の場合
        recall = recall_at_k(self.candidates_data, self.known_sites, k=2)
        # 上位2件の候補（Rio Test, Lagoa Sample）のうち、
        # Rio Testに近い遺跡1件がマッチするはず
        self.assertGreater(recall, 0)
        self.assertLessEqual(recall, 1)
        
        # K=1の場合
        recall_k1 = recall_at_k(self.candidates_data, self.known_sites, k=1)
        self.assertGreaterEqual(recall_k1, 0)
        
        # 空データの場合
        empty_df = pd.DataFrame(columns=['geometry', 'total_score'])
        recall_empty = recall_at_k(empty_df, self.known_sites, k=100)
        self.assertEqual(recall_empty, 0.0)
    
    def test_map_score(self):
        """mAP指標のテスト"""
        map_val = map_score(self.candidates_data, self.known_sites)
        self.assertGreaterEqual(map_val, 0)
        self.assertLessEqual(map_val, 1)
        
        # 空データの場合
        empty_df = pd.DataFrame(columns=['geometry', 'total_score'])
        map_empty = map_score(empty_df, self.known_sites)
        self.assertEqual(map_empty, 0.0)
    
    def test_workload(self):
        """Workload指標のテスト"""
        wl = workload(self.candidates_data)
        self.assertEqual(wl, 3)
        
        # 空データの場合
        empty_df = pd.DataFrame()
        wl_empty = workload(empty_df)
        self.assertEqual(wl_empty, 0)
    
    def test_root_diversity(self):
        """語根多様性のテスト"""
        diversity = root_diversity(self.candidates_data)
        self.assertGreater(diversity, 0)  # 3つの異なる語根があるので多様性は正の値
        
        # 同じ語根のみの場合
        same_root_data = pd.DataFrame({
            'root': ['rio', 'rio', 'rio']
        })
        diversity_same = root_diversity(same_root_data)
        self.assertEqual(diversity_same, 0)  # 多様性は0
        
        # rootカラムがない場合
        no_root_data = pd.DataFrame({
            'name': ['test1', 'test2']
        })
        diversity_no_root = root_diversity(no_root_data)
        self.assertEqual(diversity_no_root, 0.0)
    
    def test_calculate_all_metrics(self):
        """全メトリクス一括計算のテスト"""
        metrics = calculate_all_metrics(self.candidates_data, self.known_sites)
        
        # 期待されるキーが存在することを確認
        expected_keys = ['recall@50', 'recall@100', 'recall@300', 'map', 'workload', 'root_diversity']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # 値の範囲チェック
        self.assertGreaterEqual(metrics['recall@50'], 0)
        self.assertLessEqual(metrics['recall@50'], 1)
        self.assertGreaterEqual(metrics['workload'], 0)
    
    def test_analyze_spatial_distribution(self):
        """空間分布分析のテスト"""
        spatial_stats = analyze_spatial_distribution(self.candidates_data)
        
        # 期待されるキーが存在することを確認
        expected_keys = ['bbox_width', 'bbox_height', 'coord_std_x', 'coord_std_y', 'centroid_x', 'centroid_y']
        for key in expected_keys:
            self.assertIn(key, spatial_stats)
        
        # 値が数値であることを確認
        for value in spatial_stats.values():
            self.assertIsInstance(value, (int, float, np.number))


class TestAgentSchema(unittest.TestCase):
    """エージェントスキーマのテストクラス"""
    
    def test_validate_action_params(self):
        """アクションパラメータ検証のテスト"""
        # set_param アクション
        valid_set_param = {
            "param_name": "dist_threshold_km",
            "param_value": 5.0
        }
        self.assertTrue(validate_action_params("set_param", valid_set_param))
        
        invalid_set_param = {"param_name": "test"}  # param_valueが不足
        self.assertFalse(validate_action_params("set_param", invalid_set_param))
        
        # add_exclude_mask アクション
        valid_mask = {
            "mask_type": "urban",
            "mask_source": "GHSL",
            "threshold": 0.7
        }
        self.assertTrue(validate_action_params("add_exclude_mask", valid_mask))
        
        invalid_mask = {"mask_type": "urban"}  # 必要なパラメータが不足
        self.assertFalse(validate_action_params("add_exclude_mask", invalid_mask))
        
        # add_root_weight アクション
        valid_weight = {
            "root": "igarape",
            "weight": 1.5
        }
        self.assertTrue(validate_action_params("add_root_weight", valid_weight))
        
        # 無効なアクション
        self.assertFalse(validate_action_params("invalid_action", {}))
    
    def test_preset_actions(self):
        """プリセットアクションの検証"""
        for action_name, action_config in PRESET_ACTIONS.items():
            # 必要なキーが存在することを確認
            self.assertIn("action", action_config)
            self.assertIn("params", action_config)
            self.assertIn("rationale", action_config)
            
            # パラメータの妥当性を確認
            self.assertTrue(
                validate_action_params(action_config["action"], action_config["params"])
            )


class TestInspectorValidatorAgent(unittest.TestCase):
    """Inspector-Validator Agent のテストクラス"""
    
    def setUp(self):
        """テスト用データの準備"""
        # 一時ディレクトリの作成
        self.temp_dir = tempfile.mkdtemp()
        
        # テスト用CSVファイルの作成
        self.candidates_file = os.path.join(self.temp_dir, "test_candidates.csv")
        test_candidates = pd.DataFrame({
            'name': ['Test Rio', 'Test Lagoa'],
            'geometry': ['POINT (-67.0 -10.0)', 'POINT (-67.1 -10.1)'],
            'total_score': [0.9, 0.7],
            'root': ['rio', 'lagoa']
        })
        test_candidates.to_csv(self.candidates_file, index=False)
        
        # テスト用GeoPackageファイルの作成
        self.known_sites_file = os.path.join(self.temp_dir, "test_known.gpkg")
        test_known = gpd.GeoDataFrame({
            'site_id': [1, 2],
            'geometry': [Point(-67.001, -10.001), Point(-67.5, -10.5)]
        }, crs="EPSG:4326")
        test_known.to_file(self.known_sites_file, driver="GPKG")
    
    def tearDown(self):
        """テスト用ファイルのクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('tamagawa_to_z.inspector_agent.agent.OpenAI')
    def test_agent_initialization(self, mock_openai):
        """エージェント初期化のテスト"""
        # モックの設定
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_assistant = Mock()
        mock_assistant.id = "test_assistant_id"
        mock_client.beta.assistants.create.return_value = mock_assistant
        
        # エージェントの作成
        agent = InspectorValidatorAgent(api_key="test_key")
        
        # OpenAIクライアントが作成されたことを確認
        mock_openai.assert_called_once_with(api_key="test_key")
        
        # Assistantが作成されたことを確認
        mock_client.beta.assistants.create.assert_called_once()
        
        # エージェントのassistantが設定されたことを確認
        self.assertEqual(agent.assistant, mock_assistant)
    
    @patch('tamagawa_to_z.inspector_agent.agent.OpenAI')
    def test_analyze_and_propose_basic(self, mock_openai):
        """基本的な分析・提案機能のテスト"""
        # モックの設定
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Assistant作成のモック
        mock_assistant = Mock()
        mock_assistant.id = "test_assistant_id"
        mock_client.beta.assistants.create.return_value = mock_assistant
        
        # Thread作成のモック
        mock_thread = Mock()
        mock_thread.id = "test_thread_id"
        mock_client.beta.threads.create.return_value = mock_thread
        
        # Run実行のモック
        mock_run = Mock()
        mock_run.status = "completed"
        mock_client.beta.threads.runs.create_and_poll.return_value = mock_run
        
        # エージェントの作成
        agent = InspectorValidatorAgent(api_key="test_key")
        
        # 分析の実行（OpenAI APIを呼び出さないバージョン）
        results = agent.analyze_and_propose(
            candidates_path=self.candidates_file,
            known_sites_path=self.known_sites_file
        )
        
        # 基本的な結果の構造を確認
        self.assertIn("meta_info", results)
        self.assertIn("metrics", results)
        self.assertIn("spatial_stats", results)
        self.assertIn("run_id", results)
        self.assertIn("timestamp", results)
        
        # メトリクスが計算されていることを確認
        metrics = results["metrics"]
        expected_metrics = ['recall@50', 'recall@100', 'recall@300', 'map', 'workload', 'root_diversity']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
    
    def test_save_results(self):
        """結果保存機能のテスト"""
        # テスト用結果データ
        test_results = {
            "meta_info": {"region": "test_region"},
            "metrics": {"recall@100": 0.5, "workload": 100},
            "spatial_stats": {"bbox_width": 0.1},
            "diagnosis": None,
            "proposal": {
                "action": "set_param",
                "params": {"param_name": "test", "param_value": 1.0},
                "rationale": "test rationale"
            },
            "run_id": "test123",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # モックエージェントの作成（OpenAI APIを使わない）
        with patch('tamagawa_to_z.inspector_agent.agent.OpenAI'):
            agent = InspectorValidatorAgent(api_key="test_key")
            
            # 結果の保存
            output_dir = os.path.join(self.temp_dir, "output")
            agent.save_results(test_results, output_dir)
            
            # ファイルが作成されたことを確認
            self.assertTrue(os.path.exists(os.path.join(output_dir, "plan_test123.yaml")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "report_test123.md")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "results_test123.json")))
            
            # YAMLファイルの内容確認
            import yaml
            with open(os.path.join(output_dir, "plan_test123.yaml"), 'r') as f:
                plan_data = yaml.safe_load(f)
            self.assertEqual(plan_data["action"], "set_param")
            
            # JSONファイルの内容確認
            with open(os.path.join(output_dir, "results_test123.json"), 'r') as f:
                json_data = json.load(f)
            self.assertEqual(json_data["run_id"], "test123")


class TestIntegration(unittest.TestCase):
    """統合テストクラス"""
    
    def setUp(self):
        """統合テスト用データの準備"""
        self.temp_dir = tempfile.mkdtemp()
        
        # より大きなテストデータセット
        self.candidates_file = os.path.join(self.temp_dir, "integration_candidates.csv")
        candidates = pd.DataFrame({
            'name': [f'Test Site {i}' for i in range(10)],
            'geometry': [f'POINT (-67.{i} -10.{i})' for i in range(10)],
            'total_score': [0.9 - i * 0.1 for i in range(10)],
            'root': ['rio', 'lagoa', 'igarape'] * 3 + ['parana']
        })
        candidates.to_csv(self.candidates_file, index=False)
        
        # 対応する既知遺跡
        self.known_sites_file = os.path.join(self.temp_dir, "integration_known.gpkg")
        known = gpd.GeoDataFrame({
            'site_id': range(5),
            'geometry': [Point(-67 + i * 0.1, -10 + i * 0.1) for i in range(5)]
        }, crs="EPSG:4326")
        known.to_file(self.known_sites_file, driver="GPKG")
    
    def tearDown(self):
        """テスト用ファイルのクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_metrics_pipeline(self):
        """完全なメトリクス計算パイプラインのテスト"""
        # CSVファイルの読み込み
        candidates = pd.read_csv(self.candidates_file, dtype={"geometry": "string"})
        known_sites = gpd.read_file(self.known_sites_file)
        
        # メトリクス計算
        metrics = calculate_all_metrics(candidates, known_sites)
        
        # 結果の妥当性確認
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)
        
        # 各メトリクスが適切な範囲内にあることを確認
        for key, value in metrics.items():
            if key.startswith('recall@') or key == 'map':
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
            elif key == 'workload':
                self.assertGreaterEqual(value, 0)
                self.assertEqual(value, len(candidates))
            elif key == 'root_diversity':
                self.assertGreaterEqual(value, 0)


if __name__ == '__main__':
    # OpenAI APIキーが設定されていない場合の警告
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY が設定されていません。一部のテストはモックを使用して実行されます。")
    
    # テストの実行
    unittest.main(verbosity=2)