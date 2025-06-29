#!/usr/bin/env python3
"""
考古学遺跡の特徴量ベース類似度スコアリングシステム
特徴量空間での類似度計算により遺跡候補地点をスコアリング
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class ArchaeologicalSimilarityAnalyzer:
    """考古学遺跡の類似度分析システム"""
    
    def __init__(self, csv_path):
        """
        初期化
        
        Args:
            csv_path (str): 遺跡地名分析CSVファイルのパス
        """
        self.csv_path = csv_path
        self.df = None
        self.site_features = None
        self.feature_names = None
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.models = {}
        
    def load_data(self):
        """データの読み込み"""
        print("データを読み込み中...")
        self.df = pd.read_csv(self.csv_path)
        print(f"データサイズ: {self.df.shape}")
        print(f"ユニーク遺跡数: {self.df['site_name'].nunique()}")
        print("遺跡リスト:")
        for site in self.df['site_name'].unique():
            count = len(self.df[self.df['site_name'] == site])
            print(f"  - {site}: {count}件の地名")
        return self.df
    
    def extract_toponym_features(self):
        """地名カテゴリーの抽出"""
        def categorize_toponym(toponym_name):
            """地名を主要カテゴリーに分類"""
            toponym = str(toponym_name).lower()
            
            if any(x in toponym for x in ['waterway:', 'rio', 'córrego', 'igarapé']):
                return 'waterway'
            elif any(x in toponym for x in ['natural:', 'floresta', 'mata', 'campo']):
                return 'natural'
            elif any(x in toponym for x in ['place:', 'cidade', 'vila', 'povoado']):
                return 'place'
            elif any(x in toponym for x in ['highway:', 'estrada', 'rua', 'avenida', 'travessa']):
                return 'highway'
            elif any(x in toponym for x in ['landuse:', 'fazenda', 'sítio', 'área']):
                return 'landuse'
            elif any(x in toponym for x in ['man_made:', 'torre', 'ponte', 'obra']):
                return 'man_made'
            else:
                return 'other'
        
        # 地名カテゴリーを追加
        self.df['toponym_category'] = self.df['toponym_name'].apply(categorize_toponym)
        
        print("地名カテゴリー分布:")
        category_counts = self.df['toponym_category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count}件")
        
        return self.df
    
    def engineer_features(self):
        """特徴量エンジニアリング"""
        print("\n特徴量エンジニアリング開始...")
        
        # 地名カテゴリーを抽出
        self.extract_toponym_features()
        
        features_list = []
        site_names = []
        
        for site_name in self.df['site_name'].unique():
            site_data = self.df[self.df['site_name'] == site_name].copy()
            
            # 基本統計
            feature_dict = {
                'site_name': site_name,
                'total_toponyms': len(site_data),
            }
            
            # 1. 距離ベース特徴量
            feature_dict.update({
                'min_distance': site_data['radius'].min(),
                'max_distance': site_data['radius'].max(),
                'mean_distance': site_data['radius'].mean(),
                'median_distance': site_data['radius'].median(),
                'std_distance': site_data['radius'].std(),
                'q25_distance': site_data['radius'].quantile(0.25),
                'q75_distance': site_data['radius'].quantile(0.75),
            })
            
            # 2. 河川距離特徴量
            feature_dict.update({
                'min_river_distance': site_data['river_radius'].min(),
                'mean_river_distance': site_data['river_radius'].mean(),
                'median_river_distance': site_data['river_radius'].median(),
                'std_river_distance': site_data['river_radius'].std(),
            })
            
            # 3. 圏域別カウント特徴量
            for radius_threshold in [1.0, 2.0, 3.0]:
                within_radius = site_data[site_data['radius'] <= radius_threshold]
                feature_dict[f'count_within_{radius_threshold}km'] = len(within_radius)
                
                # カテゴリー別圏域カウント
                for category in ['waterway', 'natural', 'place', 'highway', 'landuse', 'man_made']:
                    count = len(within_radius[within_radius['toponym_category'] == category])
                    feature_dict[f'{category}_count_{radius_threshold}km'] = count
            
            # 4. 密度特徴量
            for radius_threshold in [1.0, 2.0, 3.0]:
                area = np.pi * radius_threshold**2
                count = len(site_data[site_data['radius'] <= radius_threshold])
                feature_dict[f'density_{radius_threshold}km'] = count / area
            
            # 5. 方位分布特徴量
            angles = site_data['angle'].values
            feature_dict.update({
                'angle_mean': np.mean(angles),
                'angle_std': np.std(angles),
                'angle_range': np.max(angles) - np.min(angles),
            })
            
            # 象限別カウント
            for i, (start, end) in enumerate([(0, 90), (90, 180), (180, 270), (270, 360)]):
                quadrant_count = len(site_data[(site_data['angle'] >= start) & (site_data['angle'] < end)])
                feature_dict[f'quadrant_{i+1}_count'] = quadrant_count
            
            # 方位均等性（エントロピー）
            quadrant_counts = [feature_dict[f'quadrant_{i+1}_count'] for i in range(4)]
            total = sum(quadrant_counts)
            if total > 0:
                probs = [c/total for c in quadrant_counts if c > 0]
                entropy = -sum(p * np.log2(p) for p in probs) if probs else 0
                feature_dict['angle_entropy'] = entropy
            else:
                feature_dict['angle_entropy'] = 0
            
            # 6. 地名タイプ比率特徴量
            for category in ['waterway', 'natural', 'place', 'highway', 'landuse', 'man_made', 'other']:
                count = len(site_data[site_data['toponym_category'] == category])
                feature_dict[f'{category}_ratio'] = count / len(site_data) if len(site_data) > 0 else 0
            
            # 7. 河川方位特徴量
            river_angles = site_data['river_angle'].values
            feature_dict.update({
                'river_angle_mean': np.mean(river_angles),
                'river_angle_std': np.std(river_angles),
            })
            
            # 8. 特殊統計量
            feature_dict.update({
                'distance_skewness': stats.skew(site_data['radius']),
                'distance_kurtosis': stats.kurtosis(site_data['radius']),
                'iqr_distance': feature_dict['q75_distance'] - feature_dict['q25_distance'],
            })
            
            features_list.append(feature_dict)
            site_names.append(site_name)
        
        # DataFrame化
        self.site_features = pd.DataFrame(features_list)
        
        # 特徴量名を保存（site_name以外）
        self.feature_names = [col for col in self.site_features.columns if col != 'site_name']
        
        print(f"生成された特徴量数: {len(self.feature_names)}")
        print(f"特徴量例: {self.feature_names[:10]}")
        
        return self.site_features
    
    def preprocess_features(self, scaler_type='standard'):
        """特徴量の前処理・正規化"""
        print(f"\n特徴量前処理（{scaler_type}スケーラー使用）...")
        
        # 数値特徴量のみ抽出
        numeric_features = self.site_features[self.feature_names].select_dtypes(include=[np.number])
        
        # 欠損値処理
        numeric_features = numeric_features.fillna(0)
        
        # スケーリング
        scaler = self.scalers[scaler_type]
        scaled_features = scaler.fit_transform(numeric_features)
        
        # DataFrame化
        self.scaled_features = pd.DataFrame(
            scaled_features, 
            columns=numeric_features.columns,
            index=self.site_features.index
        )
        
        print(f"スケーリング後の特徴量形状: {self.scaled_features.shape}")
        
        return self.scaled_features
    
    def analyze_feature_correlation(self, threshold=0.8):
        """特徴量相関分析"""
        print(f"\n特徴量相関分析（閾値: {threshold}）...")
        
        # 相関行列計算
        corr_matrix = self.scaled_features.corr().abs()
        
        # 高相関ペアを特定
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        print(f"高相関ペア数（>{threshold}）: {len(high_corr_pairs)}")
        if high_corr_pairs:
            print("高相関ペア例:")
            for pair in high_corr_pairs[:5]:
                print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        return corr_matrix, high_corr_pairs
    
    def remove_high_correlation_features(self, threshold=0.95):
        """高相関特徴量の除去"""
        corr_matrix = self.scaled_features.corr().abs()
        
        # 上三角行列から高相関ペアを見つける
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 除去すべき特徴量を特定
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        print(f"高相関により除去する特徴量数: {len(to_drop)}")
        if to_drop:
            print(f"除去特徴量例: {to_drop[:5]}")
            self.scaled_features = self.scaled_features.drop(columns=to_drop)
        
        print(f"残存特徴量数: {self.scaled_features.shape[1]}")
        return self.scaled_features
    
    def build_similarity_models(self):
        """類似度計算モデルの構築"""
        print("\n類似度計算モデル構築中...")
        
        X = self.scaled_features.values
        
        # 1. k最近傍モデル
        self.models['knn'] = NearestNeighbors(n_neighbors=3, metric='euclidean')
        self.models['knn'].fit(X)
        
        # 2. クラスタリングモデル
        self.models['kmeans'] = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.models['kmeans'].fit(X)
        
        self.models['gaussian_mixture'] = GaussianMixture(n_components=3, random_state=42)
        self.models['gaussian_mixture'].fit(X)
        
        # 3. 異常検知モデル
        self.models['isolation_forest'] = IsolationForest(contamination=0.2, random_state=42)
        self.models['isolation_forest'].fit(X)
        
        self.models['one_class_svm'] = OneClassSVM(nu=0.2)
        self.models['one_class_svm'].fit(X)
        
        print(f"構築されたモデル数: {len(self.models)}")
        print(f"モデル種類: {list(self.models.keys())}")
        
        return self.models
    
    def calculate_similarity_scores(self, candidate_features=None):
        """類似度スコアの計算"""
        print("\n類似度スコア計算中...")
        
        if candidate_features is None:
            # 既知遺跡に対するスコア計算（Leave-One-Out方式）
            X = self.scaled_features.values
            scores = []
            
            for i in range(len(X)):
                # 一つを除いた訓練データ
                train_X = np.delete(X, i, axis=0)
                test_X = X[i].reshape(1, -1)
                
                # 各モデルでスコア計算
                score_dict = self._calculate_single_score(train_X, test_X)
                score_dict['site_name'] = self.site_features.iloc[i]['site_name']
                scores.append(score_dict)
            
            return pd.DataFrame(scores)
        else:
            # 新候補地点に対するスコア計算
            X_train = self.scaled_features.values
            X_test = candidate_features
            
            score_dict = self._calculate_single_score(X_train, X_test)
            return score_dict
    
    def _calculate_single_score(self, X_train, X_test):
        """単一地点の類似度スコア計算"""
        scores = {}
        
        # 1. k最近傍距離スコア
        knn_temp = NearestNeighbors(n_neighbors=min(3, len(X_train)), metric='euclidean')
        knn_temp.fit(X_train)
        distances, _ = knn_temp.kneighbors(X_test)
        knn_score = np.exp(-distances.mean())
        scores['knn_score'] = knn_score
        
        # 2. クラスタ類似度スコア
        kmeans_temp = KMeans(n_clusters=min(3, len(X_train)), random_state=42, n_init=10)
        kmeans_temp.fit(X_train)
        cluster_distances = []
        for center in kmeans_temp.cluster_centers_:
            dist = np.linalg.norm(X_test - center.reshape(1, -1), axis=1)
            cluster_distances.append(dist[0])
        cluster_score = np.exp(-min(cluster_distances))
        scores['cluster_score'] = cluster_score
        
        # 3. ガウシアン混合モデルスコア
        if len(X_train) >= 3:
            gm_temp = GaussianMixture(n_components=min(3, len(X_train)), random_state=42)
            gm_temp.fit(X_train)
            gm_score = np.exp(gm_temp.score(X_test))
            scores['gaussian_score'] = gm_score
        else:
            scores['gaussian_score'] = knn_score  # fallback
        
        # 4. 異常度スコア（低いほど類似）
        iso_temp = IsolationForest(contamination=0.2, random_state=42)
        iso_temp.fit(X_train)
        anomaly_score = iso_temp.decision_function(X_test)[0]
        # 正規化して類似度に変換
        anomaly_similarity = (anomaly_score + 1) / 2  # [-1,1] -> [0,1]
        scores['anomaly_score'] = anomaly_similarity
        
        # 5. 複合スコア
        composite_score = (
            scores['knn_score'] * 0.35 +
            scores['cluster_score'] * 0.25 +
            scores['gaussian_score'] * 0.25 +
            scores['anomaly_score'] * 0.15
        )
        scores['composite_score'] = composite_score
        
        return scores
    
    def evaluate_models(self):
        """モデル評価（交差検証）"""
        print("\nモデル評価中...")
        
        X = self.scaled_features.values
        
        # シルエット係数
        if len(X) > 1:
            kmeans_labels = self.models['kmeans'].labels_
            silhouette = silhouette_score(X, kmeans_labels)
            
            calinski = calinski_harabasz_score(X, kmeans_labels)
            davies_bouldin = davies_bouldin_score(X, kmeans_labels)
            
            print(f"シルエット係数: {silhouette:.3f}")
            print(f"Calinski-Harabasz指数: {calinski:.3f}")
            print(f"Davies-Bouldin指数: {davies_bouldin:.3f}")
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies_bouldin
            }
        else:
            print("評価には最低2つのサンプルが必要です")
            return {}
    
    def visualize_features(self, method='pca'):
        """特徴量空間の可視化"""
        print(f"\n特徴量空間可視化（{method}）...")
        
        X = self.scaled_features.values
        site_names = self.site_features['site_name'].values
        
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1))
        else:
            raise ValueError("サポートされていない手法です")
        
        X_reduced = reducer.fit_transform(X)
        
        # プロット
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                             c=range(len(X)), cmap='tab10', s=100, alpha=0.7)
        
        # サイト名をラベル表示
        for i, name in enumerate(site_names):
            plt.annotate(name, (X_reduced[i, 0], X_reduced[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        plt.title(f'Archaeological Sites in Feature Space ({method.upper()})')
        plt.xlabel(f'{method.upper()}-1')
        plt.ylabel(f'{method.upper()}-2')
        plt.grid(True, alpha=0.3)
        
        # カラーバー
        cbar = plt.colorbar(scatter)
        cbar.set_label('Site Index')
        
        plt.tight_layout()
        plt.savefig(f'archaeological_sites_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if method == 'pca':
            print(f"PCA第1主成分の寄与率: {reducer.explained_variance_ratio_[0]:.3f}")
            print(f"PCA第2主成分の寄与率: {reducer.explained_variance_ratio_[1]:.3f}")
            print(f"累積寄与率: {reducer.explained_variance_ratio_.sum():.3f}")
        
        return X_reduced, reducer
    
    def generate_analysis_report(self, scores_df):
        """類似度分析レポートを生成"""
        report = []
        report.append("# 考古学遺跡類似度分析レポート")
        report.append("")
        report.append("## 分析概要")
        report.append("既知考古学遺跡の地名分布パターンを機械学習で分析し、類似度スコアを算出しました。")
        report.append("")
        
        report.append("## 遺跡類似度スコア")
        report.append("| 順位 | 遺跡名 | 複合スコア | kNN | クラスタ | ガウシアン | 異常度 |")
        report.append("|------|--------|------------|-----|-----------|------------|--------|")
        
        sorted_scores = scores_df.sort_values('composite_score', ascending=False)
        for i, (_, row) in enumerate(sorted_scores.iterrows(), 1):
            report.append(f"| {i} | {row['site_name']} | {row['composite_score']:.4f} | {row['knn_score']:.4f} | {row['cluster_score']:.4f} | {row['gaussian_score']:.4f} | {row['anomaly_score']:.4f} |")
        
        report.append("")
        report.append("## 統計サマリー")
        report.append(f"- **平均複合スコア**: {scores_df['composite_score'].mean():.4f}")
        report.append(f"- **標準偏差**: {scores_df['composite_score'].std():.4f}")
        report.append(f"- **最高スコア**: {scores_df['composite_score'].max():.4f}")
        report.append(f"- **最低スコア**: {scores_df['composite_score'].min():.4f}")
        report.append("")
        
        report.append("## 使用した機械学習手法")
        report.append("1. **k最近傍法**: 特徴量空間での近傍距離による類似度")
        report.append("2. **クラスタリング**: K-meansクラスタ中心からの距離")
        report.append("3. **ガウシアン混合**: 確率分布による適合度")
        report.append("4. **異常検知**: Isolation Forestによる正常度評価")
        report.append("")
        
        report.append("複合スコア = kNN×0.35 + クラスタ×0.25 + ガウシアン×0.25 + 異常度×0.15")
        
        return "\n".join(report)


def main():
    """メイン実行関数"""
    # CSVファイルパス
    csv_path = "/Users/kisamorikeiichi/Development/tamagawa_to_z/data/output/site_analysis/site_analysis_20250627_001252/site_toponym_analysis_acre.csv"
    
    # 分析器初期化
    analyzer = ArchaeologicalSimilarityAnalyzer(csv_path)
    
    print("=== 考古学遺跡類似度分析システム ===")
    
    # ステップ1: データ読み込み
    df = analyzer.load_data()
    
    # ステップ2: 特徴量エンジニアリング
    site_features = analyzer.engineer_features()
    
    # ステップ3: 前処理
    scaled_features = analyzer.preprocess_features()
    
    # ステップ4: 相関分析
    corr_matrix, high_corr_pairs = analyzer.analyze_feature_correlation()
    
    # ステップ5: 高相関特徴量除去
    cleaned_features = analyzer.remove_high_correlation_features()
    
    # ステップ6: 類似度モデル構築
    models = analyzer.build_similarity_models()
    
    # ステップ7: モデル評価
    evaluation_metrics = analyzer.evaluate_models()
    
    # ステップ8: 類似度スコア計算
    similarity_scores = analyzer.calculate_similarity_scores()
    
    # ステップ9: 結果表示
    print("\n=== 遺跡類似度スコア結果 ===")
    print(similarity_scores.round(4))
    
    # 複合スコアでソート
    top_sites = similarity_scores.sort_values('composite_score', ascending=False)
    print("\n=== 複合スコア上位遺跡 ===")
    for _, row in top_sites.iterrows():
        print(f"{row['site_name']}: {row['composite_score']:.4f}")
    
    # ステップ10: 可視化（スキップ - GUI環境でないため）
    print("\n可視化はスキップしました（コマンドライン環境）")
    
    # ステップ11: 統計サマリー
    print("\n=== 分析統計サマリー ===")
    print(f"・分析対象遺跡数: {len(site_features)}")
    print(f"・最終特徴量数: {cleaned_features.shape[1]}")
    print(f"・除去された高相関特徴量数: {len(analyzer.feature_names) - cleaned_features.shape[1]}")
    
    if evaluation_metrics:
        print(f"・クラスタリング評価:")
        for metric, value in evaluation_metrics.items():
            print(f"  - {metric}: {value:.4f}")
    
    print(f"・平均複合スコア: {similarity_scores['composite_score'].mean():.4f}")
    print(f"・複合スコア標準偏差: {similarity_scores['composite_score'].std():.4f}")
    
    # 結果をCSVで保存
    output_path = "archaeological_similarity_scores.csv"
    similarity_scores.to_csv(output_path, index=False)
    print(f"\n結果を {output_path} に保存しました")
    
    return analyzer, similarity_scores


if __name__ == "__main__":
    analyzer = main()