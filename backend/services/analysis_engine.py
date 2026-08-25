import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, silhouette_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)

# Global styling configuration for publication-ready visual plots
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.titlesize': 12,
    'figure.dpi': 150
})

class AnalysisEngine:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        # Exclude CSV index artifacts and columns that are only row numbers.
        index_cols = []
        for column in self.df.columns:
            values = self.df[column].dropna()
            normalized_name = str(column).strip().lower()
            is_named_index = normalized_name == 'id' or 'unnamed' in normalized_name
            is_serial = (
                pd.api.types.is_numeric_dtype(self.df[column])
                and len(values) == len(self.df)
                and len(values) > 1
                and values.is_unique
                and values.iloc[0] in (0, 1)
                and np.array_equal(values.to_numpy(), np.arange(values.iloc[0], values.iloc[0] + len(values)))
            )
            if is_named_index or is_serial:
                index_cols.append(column)
        if index_cols:
            self.df = self.df.drop(columns=index_cols)

    def run_full_analysis(self, output_dir: str = "backend/uploads"):
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # -------------------------------------------------------------
        # 1. Dataset Overview
        # -------------------------------------------------------------
        rows = len(self.df)
        cols = len(self.df.columns)
        total_cells = max(rows * cols, 1)
        total_nulls = int(self.df.isnull().sum().sum())
        missing_pct = float((total_nulls / total_cells) * 100)
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_pct = float((duplicate_rows / max(rows, 1)) * 100)
        memory_mb = float(self.df.memory_usage(deep=True).sum() / (1024 * 1024))
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        results['overview'] = {
            'rows': rows,
            'cols': cols,
            'missing_pct': round(missing_pct, 2),
            'total_nulls': total_nulls,
            'duplicate_rows': duplicate_rows,
            'duplicate_pct': round(duplicate_pct, 2),
            'memory_mb': round(memory_mb, 2),
            'num_cols_count': len(num_cols),
            'cat_cols_count': len(cat_cols)
        }
        # Backward-compatibility keys
        results['rows'] = rows
        results['cols'] = cols
        results['missing_pct'] = missing_pct

        # -------------------------------------------------------------
        # 2. Top 5 Columns by Missing Values (Numeric Columns list)
        # -------------------------------------------------------------
        missing_series = self.df.isnull().sum()
        missing_df = pd.DataFrame({
            'column': missing_series.index,
            'missing_count': missing_series.values,
            'missing_pct': (missing_series.values / max(rows, 1)) * 100
        }).sort_values(by='missing_count', ascending=False)
        
        top5_missing = missing_df.head(5).to_dict(orient='records')
        for item in top5_missing:
            item['missing_pct'] = round(float(item['missing_pct']), 2)
            item['missing_count'] = int(item['missing_count'])
            
        results['top5_missing'] = top5_missing
        results['numeric_columns_list'] = num_cols
        results['categorical_columns_list'] = cat_cols
        results['num_cols'] = num_cols
        results['cat_cols'] = cat_cols

        # -------------------------------------------------------------
        # 3. Numeric Summary (chunked into features 1-5, 6-10, 11-15, etc.)
        # -------------------------------------------------------------
        numeric_summaries = {}
        if len(num_cols) > 0:
            desc_df = self.df[num_cols].describe().T
            desc_df['skewness'] = self.df[num_cols].skew()
            desc_df['kurtosis'] = self.df[num_cols].kurtosis()
            desc_df['missing_count'] = self.df[num_cols].isnull().sum()
            desc_df['missing_pct'] = (desc_df['missing_count'] / max(rows, 1)) * 100
            
            chunk_size = 5
            total_chunks = max(6, math.ceil(len(num_cols) / chunk_size))
            
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                label = f"features {start_idx + 1}–{end_idx}"
                
                chunk_cols = num_cols[start_idx:end_idx]
                if chunk_cols:
                    chunk_stats = desc_df.loc[chunk_cols].reset_index()
                    chunk_stats.rename(columns={'index': 'feature'}, inplace=True)
                    # Round numeric fields
                    for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis', 'missing_pct']:
                        if col in chunk_stats.columns:
                            chunk_stats[col] = chunk_stats[col].apply(lambda x: round(float(x), 3) if pd.notnull(x) else 0.0)
                    chunk_stats['count'] = chunk_stats['count'].astype(int)
                    chunk_stats['missing_count'] = chunk_stats['missing_count'].astype(int)
                    numeric_summaries[label] = chunk_stats.to_dict(orient='records')
                else:
                    numeric_summaries[label] = []
                    
        results['numeric_summaries'] = numeric_summaries

        # -------------------------------------------------------------
        # 4. Correlation Heatmap
        # -------------------------------------------------------------
        if len(num_cols) >= 2:
            corr_matrix = self.df[num_cols].corr()
            plt.figure(figsize=(10, 8))
            annot = len(num_cols) <= 15
            sns.heatmap(
                corr_matrix,
                cmap='coolwarm',
                center=0,
                annot=annot,
                fmt=".2f" if annot else "",
                cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=0.5 if len(num_cols) <= 20 else 0,
                square=True
            )
            plt.title("Correlation Heatmap", fontsize=14, pad=12, fontweight='bold', color="#1A365D")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            corr_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.tight_layout()
            plt.savefig(corr_path, dpi=200)
            plt.close()
            results['correlation_heatmap_plot'] = corr_path
        else:
            results['correlation_heatmap_plot'] = None

        # -------------------------------------------------------------
        # 5. Categorical Columns — Value counts: diagnosis (or detected target)
        # -------------------------------------------------------------
        # Identify target column: prefer column named 'diagnosis', 'target', 'label', 'class', or binary categorical
        target_col = None
        for col_cand in ['diagnosis', 'Diagnosis', 'target', 'Target', 'label', 'Label', 'class', 'Class']:
            if col_cand in self.df.columns:
                target_col = col_cand
                break
                
        if not target_col:
            for col in cat_cols + num_cols:
                if self.df[col].nunique() == 2:
                    target_col = col
                    break
        if not target_col and len(cat_cols) > 0:
            target_col = cat_cols[0]
            
        results['target_col'] = target_col
        categorical_value_counts = {}
        
        target_focus_col = target_col if (target_col and target_col in cat_cols) else (cat_cols[0] if cat_cols else None)
        if target_focus_col:
            vc = self.df[target_focus_col].value_counts(dropna=False)
            vp = self.df[target_focus_col].value_counts(normalize=True, dropna=False) * 100
            vc_list = []
            for category, count in vc.items():
                vc_list.append({
                    'category': str(category),
                    'count': int(count),
                    'percentage': round(float(vp[category]), 2)
                })
            categorical_value_counts[target_focus_col] = vc_list
            results['categorical_value_counts'] = categorical_value_counts
            results['primary_categorical_target'] = target_focus_col
            results['diagnosis_value_counts'] = vc_list
        else:
            results['categorical_value_counts'] = {}
            results['diagnosis_value_counts'] = []

        # -------------------------------------------------------------
        # 6. Distribution of diagnosis / Target
        # -------------------------------------------------------------
        if target_col and target_col in self.df.columns:
            plt.figure(figsize=(7, 4.5))
            target_series = self.df[target_col].astype(str)
            counts = target_series.value_counts()
            ax = plt.gca()
            bars = ax.bar(range(len(counts)), counts.values, color=plt.cm.Set2(np.linspace(0, 1, len(counts))))
            ax.set_xticks(range(len(counts)), counts.index)
            total_cnt = len(target_series)
            for bar, cnt in zip(bars, counts.values):
                cnt = int(cnt)
                pct = (cnt / total_cnt) * 100
                ax.annotate(f'{cnt:,}\n({pct:.1f}%)',
                            (bar.get_x() + bar.get_width() / 2., bar.get_height() / 2.),
                            ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                            
            plt.title(f"Distribution of {target_col}", fontsize=13, pad=10, fontweight='bold', color="#1A365D")
            plt.xlabel(str(target_col), fontsize=10, fontweight='bold')
            plt.ylabel("Count", fontsize=10, fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            target_dist_path = os.path.join(output_dir, "distribution_of_diagnosis.png")
            plt.tight_layout()
            plt.savefig(target_dist_path, dpi=200)
            plt.close()
            results['target_distribution_plot'] = target_dist_path
        else:
            results['target_distribution_plot'] = None

        # -------------------------------------------------------------
        # 7. KDE Plots – Pages 1 to 5
        # -------------------------------------------------------------
        kde_plot_paths = []
        if len(num_cols) > 0:
            # Group into 5 pages (e.g. 6 features per page for standard 30 features)
            features_per_page = max(1, math.ceil(min(len(num_cols), 30) / 5)) if len(num_cols) <= 30 else 6
            plot_df = self.df.sample(n=min(len(self.df), 1000), random_state=42)
            for page in range(1, 6):
                start_f = (page - 1) * features_per_page
                end_f = start_f + features_per_page
                page_features = num_cols[start_f:end_f]
                
                if page_features:
                    cols_in_grid = 3 if len(page_features) > 2 else len(page_features)
                    rows_in_grid = math.ceil(len(page_features) / cols_in_grid)
                    fig, axes = plt.subplots(rows_in_grid, cols_in_grid, figsize=(11, 3.2 * rows_in_grid))
                    if not isinstance(axes, np.ndarray):
                        axes = np.array([axes])
                    axes_flat = axes.flatten()
                    
                    for idx, feat in enumerate(page_features):
                        ax = axes_flat[idx]
                        feat_data = plot_df[[feat]].dropna()
                        if target_col and target_col in self.df.columns and self.df[target_col].nunique() <= 5:
                            hue_data = plot_df[[feat, target_col]].dropna()
                            for class_index, (class_name, class_data) in enumerate(hue_data.groupby(target_col, sort=False)):
                                ax.hist(class_data[feat], bins=30, density=True, alpha=0.35,
                                        histtype="stepfilled", label=str(class_name))
                            ax.legend(fontsize=7)
                        else:
                            ax.hist(feat_data[feat], bins=30, density=True, color="#2B6CB0",
                                    alpha=0.4, histtype="stepfilled")
                        ax.set_title(f"{feat}", fontsize=9.5, fontweight='bold', color="#2D3748")
                        ax.set_xlabel("")
                        ax.grid(True, linestyle='--', alpha=0.4)
                        
                    # Hide unused axes
                    for idx in range(len(page_features), len(axes_flat)):
                        axes_flat[idx].set_visible(False)
                        
                    fig.suptitle(f"KDE Density Distributions – Page {page}", fontsize=13, fontweight='bold', color="#1A365D", y=1.02)
                    plt.tight_layout()
                    kde_path = os.path.join(output_dir, f"kde_page_{page}.png")
                    plt.savefig(kde_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    kde_plot_paths.append(kde_path)
                    results[f'kde_page_{page}_plot'] = kde_path
                else:
                    results[f'kde_page_{page}_plot'] = None
        results['kde_plot_paths'] = kde_plot_paths

        # -------------------------------------------------------------
        # 8. Outlier Proportion per Column (IQR Method)
        # -------------------------------------------------------------
        outlier_data = []
        if len(num_cols) > 0:
            for col in num_cols:
                series = self.df[col].dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_pct = (outlier_count / len(series)) * 100
                    outlier_data.append({
                        'column': col,
                        'outlier_count': int(outlier_count),
                        'outlier_pct': round(float(outlier_pct), 2),
                        'lower_bound': round(float(lower_bound), 3),
                        'upper_bound': round(float(upper_bound), 3)
                    })
            outlier_df = pd.DataFrame(outlier_data).sort_values(by='outlier_pct', ascending=True)
            results['outlier_proportions'] = outlier_df.to_dict(orient='records')
            
            # Plot Outlier Bar Chart
            plt.figure(figsize=(9, max(4, len(num_cols) * 0.22)))
            plot_df = outlier_df.tail(20) if len(outlier_df) > 20 else outlier_df
            sns.barplot(data=plot_df, x='outlier_pct', y='column', palette='Reds_d')
            plt.title("Outlier Proportion per Column (IQR Method: 1.5 * IQR)", fontsize=12, fontweight='bold', color="#1A365D")
            plt.xlabel("Outlier Percentage (%)", fontsize=10, fontweight='bold')
            plt.ylabel("Features", fontsize=10, fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            outlier_path = os.path.join(output_dir, "outlier_distribution.png")
            plt.tight_layout()
            plt.savefig(outlier_path, dpi=200)
            plt.close()
            results['outlier_plot'] = outlier_path
        else:
            results['outlier_proportions'] = []
            results['outlier_plot'] = None

        # -------------------------------------------------------------
        # 9. Duplicate Distribution per Column
        # -------------------------------------------------------------
        dup_col_data = []
        for col in self.df.columns:
            non_unique = len(self.df) - self.df[col].nunique(dropna=False)
            dup_pct = (non_unique / max(len(self.df), 1)) * 100
            dup_col_data.append({
                'column': col,
                'duplicate_pct': round(float(dup_pct), 2),
                'unique_count': int(self.df[col].nunique())
            })
        dup_df = pd.DataFrame(dup_col_data).sort_values(by='duplicate_pct', ascending=True)
        results['duplicate_distribution'] = dup_df.to_dict(orient='records')
        
        plt.figure(figsize=(9, max(4, len(self.df.columns) * 0.22)))
        plot_dup_df = dup_df.tail(20) if len(dup_df) > 20 else dup_df
        sns.barplot(data=plot_dup_df, x='duplicate_pct', y='column', palette='Purples_d')
        plt.title("Duplicate / Non-Unique Value Ratio per Column", fontsize=12, fontweight='bold', color="#1A365D")
        plt.xlabel("Non-Unique Values (%)", fontsize=10, fontweight='bold')
        plt.ylabel("Columns", fontsize=10, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        dup_path = os.path.join(output_dir, "duplicate_distribution.png")
        plt.tight_layout()
        plt.savefig(dup_path, dpi=200)
        plt.close()
        results['duplicate_plot'] = dup_path

        # -------------------------------------------------------------
        # 10 & 13. ML Modeling: Feature Importance, ROC/Residuals, Metrics
        # -------------------------------------------------------------
        if target_col and len(num_cols) > 0:
            features = [c for c in num_cols if c != target_col]
            if len(features) > 0:
                X = self.df[features].fillna(self.df[features].median())
                target_raw = self.df[target_col]
                
                # Check if Classification or Regression
                is_classification = target_raw.nunique() <= 10 or target_raw.dtype == 'object'
                
                if is_classification:
                    y, class_names = pd.factorize(target_raw)
                    results['is_classification'] = True
                    results['class_names'] = [str(c) for c in class_names]
                    
                    if len(np.unique(y)) >= 2:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                        )
                        clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        clf.fit(X_train, y_train)
                        
                        preds = clf.predict(X_test)
                        probs = clf.predict_proba(X_test)
                        
                        results['accuracy'] = float(accuracy_score(y_test, preds))
                        results['precision'] = float(precision_score(y_test, preds, average='weighted', zero_division=0))
                        results['recall'] = float(recall_score(y_test, preds, average='weighted', zero_division=0))
                        results['f1_score'] = float(f1_score(y_test, preds, average='weighted', zero_division=0))
                        
                        # Feature Importance
                        importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False).head(15)
                        results['feature_importance_ranking'] = [
                            {'feature': k, 'importance': round(float(v), 4)} for k, v in importances.items()
                        ]
                        
                        plt.figure(figsize=(8, 5))
                        sns.barplot(x=importances.values, y=importances.index, palette="Blues_r")
                        plt.title(f"Feature Importance (Random Forest – Target: {target_col})", fontsize=12, fontweight='bold', color="#1A365D")
                        plt.xlabel("Gini Importance Score", fontsize=10, fontweight='bold')
                        plt.grid(axis='x', linestyle='--', alpha=0.5)
                        feat_path = os.path.join(output_dir, "feature_importance.png")
                        plt.tight_layout()
                        plt.savefig(feat_path, dpi=200)
                        plt.close()
                        results['feature_importance_plot'] = feat_path
                        
                        # ROC Curve (for binary classification)
                        if len(np.unique(y)) == 2:
                            prob_pos = probs[:, 1]
                            auc_val = float(roc_auc_score(y_test, prob_pos))
                            results['roc_auc'] = auc_val
                            
                            fpr, tpr, _ = roc_curve(y_test, prob_pos)
                            plt.figure(figsize=(6, 4.5))
                            plt.plot(fpr, tpr, color='#1E40AF', lw=2.5, label=f"ROC (AUC = {auc_val:.4f})")
                            plt.plot([0, 1], [0, 1], color='#6B7280', lw=1.5, linestyle='--')
                            plt.fill_between(fpr, tpr, alpha=0.15, color='#3B82F6')
                            plt.xlim([-0.02, 1.02])
                            plt.ylim([-0.02, 1.05])
                            plt.xlabel("False Positive Rate", fontsize=10, fontweight='bold')
                            plt.ylabel("True Positive Rate", fontsize=10, fontweight='bold')
                            plt.title("Model Summary & ROC Curve", fontsize=12, fontweight='bold', color="#1A365D")
                            plt.legend(loc="lower right", frameon=True)
                            plt.grid(True, linestyle='--', alpha=0.4)
                            roc_path = os.path.join(output_dir, "roc_curve.png")
                            plt.tight_layout()
                            plt.savefig(roc_path, dpi=200)
                            plt.close()
                            results['roc_plot'] = roc_path
                            
                            # 12. Predicted Probability vs Actual
                            plt.figure(figsize=(7, 4.5))
                            prob_df = pd.DataFrame({
                                'Predicted_Probability': prob_pos,
                                'Actual_Class': [str(class_names[c]) for c in y_test]
                            })
                            for class_index, (class_name, class_data) in enumerate(
                                prob_df.groupby('Actual_Class', sort=False)
                            ):
                                plt.hist(class_data['Predicted_Probability'], bins=15, alpha=0.5,
                                         density=False, histtype='stepfilled', label=class_name)
                            plt.legend(fontsize=8)
                            plt.title("Predicted Probability vs Actual Class Distribution", fontsize=12, fontweight='bold', color="#1A365D")
                            label_pos = str(class_names[1]) if len(class_names) > 1 else "Positive Class"
                            plt.xlabel(f"Predicted Probability of {label_pos}", fontsize=10, fontweight='bold')
                            plt.ylabel("Frequency", fontsize=10, fontweight='bold')
                            plt.grid(True, linestyle='--', alpha=0.4)
                            pred_prob_path = os.path.join(output_dir, "predicted_probability_vs_actual.png")
                            plt.tight_layout()
                            plt.savefig(pred_prob_path, dpi=200)
                            plt.close()
                            results['pred_prob_vs_actual_plot'] = pred_prob_path
                            
                        # Confusion Matrix
                        cm = confusion_matrix(y_test, preds)
                        plt.figure(figsize=(5.5, 4.5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=[str(c) for c in class_names],
                                    yticklabels=[str(c) for c in class_names])
                        plt.title("Confusion Matrix", fontsize=12, fontweight='bold', color="#1A365D")
                        plt.xlabel("Predicted Class", fontsize=10, fontweight='bold')
                        plt.ylabel("Actual Class", fontsize=10, fontweight='bold')
                        cm_path = os.path.join(output_dir, "confusion_matrix.png")
                        plt.tight_layout()
                        plt.savefig(cm_path, dpi=200)
                        plt.close()
                        results['confusion_matrix_plot'] = cm_path
                else:
                    # Regression
                    results['is_classification'] = False
                    y = target_raw.values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    reg = RandomForestRegressor(n_estimators=100, random_state=42)
                    reg.fit(X_train, y_train)
                    preds = reg.predict(X_test)
                    
                    results['r2_score'] = float(r2_score(y_test, preds))
                    results['rmse'] = float(np.sqrt(mean_squared_error(y_test, preds)))
                    results['mae'] = float(mean_absolute_error(y_test, preds))
                    
                    # Residuals Plot
                    residuals = y_test - preds
                    plt.figure(figsize=(7, 4.5))
                    plt.scatter(preds, residuals, alpha=0.6, color="#2563EB", edgecolors='none')
                    plt.axhline(0, color='red', linestyle='--', lw=1.5)
                    plt.title(f"Residuals Plot (R² = {results['r2_score']:.4f})", fontsize=12, fontweight='bold', color="#1A365D")
                    plt.xlabel("Predicted Values", fontsize=10, fontweight='bold')
                    plt.ylabel("Residuals (Actual - Predicted)", fontsize=10, fontweight='bold')
                    plt.grid(True, linestyle='--', alpha=0.4)
                    res_path = os.path.join(output_dir, "residuals_plot.png")
                    plt.tight_layout()
                    plt.savefig(res_path, dpi=200)
                    plt.close()
                    results['roc_plot'] = res_path
                    results['pred_prob_vs_actual_plot'] = res_path

        # -------------------------------------------------------------
        # 11. PCA Clusters & Silhouette Score
        # -------------------------------------------------------------
        if len(num_cols) >= 2:
            X_num = self.df[num_cols].fillna(self.df[num_cols].median())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_num)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(components)
            sil_score = float(silhouette_score(components, kmeans.labels_))
            results['silhouette_score'] = sil_score
            
            plt.figure(figsize=(7.5, 5))
            scatter = plt.scatter(
                components[:, 0], components[:, 1],
                c=kmeans.labels_, cmap="coolwarm", alpha=0.7, s=25, edgecolors='none'
            )
            # Plot centroids
            centroids = kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=120, label='Centroids')
            
            plt.title(f"PCA Clusters (Silhouette = {sil_score:.3f})", fontsize=12, fontweight='bold', color="#1A365D")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=10, fontweight='bold')
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=10, fontweight='bold')
            plt.legend(frameon=True, loc="best")
            plt.grid(True, linestyle='--', alpha=0.4)
            pca_path = os.path.join(output_dir, "pca_clusters.png")
            plt.tight_layout()
            plt.savefig(pca_path, dpi=200)
            plt.close()
            results['pca_plot'] = pca_path
        else:
            results['silhouette_score'] = 0.0
            results['pca_plot'] = None

        return results