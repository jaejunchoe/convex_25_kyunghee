import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy as cp
from datetime import datetime
import time
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    df = pd.read_csv(filepath)
    df_processed = df[['주차장명', '위도', '경도', '총 주차면', '현재 주차 차량수', '기본 주차 요금', '기본 주차 시간(분 단위)']].copy()
    df_processed.columns = ['name', 'lat', 'lon', 'capacity', 'current_count', 'basic_fee', 'basic_time_min']
    df_processed['basic_time_min'] = df_processed['basic_time_min'].replace(0, 5) 
    df_processed['fee_per_min'] = df_processed['basic_fee'] / df_processed['basic_time_min']
    df_processed['fee_per_min'] = df_processed['fee_per_min'].fillna(0)
    df_processed['capacity'] = df_processed['capacity'].fillna(0).astype(int)
    df_processed['current_count'] = df_processed['current_count'].fillna(0).astype(int)
    df_processed = df_processed.dropna(subset=['lat', 'lon'])
    return df_processed

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    return c * 6371000

def gini_coefficient(values):
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n

def generate_users(n_users, parking_df):
    min_lat, max_lat = parking_df['lat'].min(), parking_df['lat'].max()
    min_lon, max_lon = parking_df['lon'].min(), parking_df['lon'].max()
    lat_buffer = (max_lat - min_lat) * 0.1
    lon_buffer = (max_lon - min_lon) * 0.1
    user_lats = np.random.uniform(min_lat - lat_buffer, max_lat + lat_buffer, n_users)
    user_lons = np.random.uniform(min_lon - lon_buffer, max_lon + lon_buffer, n_users)
    user_times = np.random.uniform(1, 4, n_users)
    return {'lat': user_lats, 'lon': user_lons, 'time': user_times}

def calculate_distance_matrix(users, parking_df):
    n_users = len(users['lat'])
    n_parkings = len(parking_df)
    d_matrix = np.zeros((n_users, n_parkings))
    parking_lats = parking_df['lat'].values
    parking_lons = parking_df['lon'].values
    for i in range(n_users):
        d_matrix[i, :] = haversine(users['lat'][i], users['lon'][i], parking_lats, parking_lons)
    return d_matrix

def run_random(users, parking_df, d_matrix):
    n_users = len(users['lat'])
    current_capacity = parking_df['capacity'].values - parking_df['current_count'].values
    current_capacity = np.maximum(current_capacity, 0)
    assignments = np.full(n_users, -1)
    available_lots = np.where(current_capacity > 0)[0]
    for i in range(n_users):
        if len(available_lots) == 0:
            break
        chosen = np.random.choice(available_lots)
        assignments[i] = chosen
        current_capacity[chosen] -= 1
        if current_capacity[chosen] == 0:
            available_lots = available_lots[available_lots != chosen]
    return assignments

def run_greedy(users, parking_df, d_matrix):
    n_users = len(users['lat'])
    current_capacity = parking_df['capacity'].values - parking_df['current_count'].values
    current_capacity = np.maximum(current_capacity, 0)
    assignments = np.full(n_users, -1)
    for i in range(n_users):
        dists = d_matrix[i, :]
        sorted_indices = np.argsort(dists)
        for p_idx in sorted_indices:
            if current_capacity[p_idx] > 0:
                assignments[i] = p_idx
                current_capacity[p_idx] -= 1
                break
    return assignments

def run_convex(users, parking_df, d_matrix, weights=(0.8, 0.1, 0.1)):
    """Convex Optimization without distance constraint"""
    alpha, beta, gamma = weights
    n_users = len(users['lat'])
    n_parkings = len(parking_df)
    
    C = parking_df['capacity'].values
    c = parking_df['current_count'].values
    p = parking_df['fee_per_min'].values * 60
    t = users['time']
    
    x = cp.Variable((n_users, n_parkings), nonneg=True)
    
    C_safe = np.maximum(C, 1) 
    congestion_terms = []
    for j in range(n_parkings):
        term = cp.square((c[j] + cp.sum(x[:, j])) / C_safe[j])
        congestion_terms.append(term)
        
    confusion = cp.sum(congestion_terms)
    distance = cp.sum(cp.multiply(d_matrix, x)) / 1000.0
    cost_matrix = np.outer(t, p)
    cost = cp.sum(cp.multiply(cost_matrix, x)) / 10000.0 
    
    objective = cp.Minimize(alpha * confusion + beta * distance + gamma * cost)
    
    constraints = [cp.sum(x, axis=1) == 1, x <= 1]
    
    available_capacity = np.maximum(C - c, 0)
    
    # 혼잡도 상한 제약: (cⱼ + Σᵢ xᵢⱼ) / Cⱼ ≤ ρ_max
    # 단, 이미 이용률이 ρ_max를 초과하는 주차장은 제외
    rho_max = 0.95  # 최대 이용률 95%
    
    for j in range(n_parkings):
        current_utilization = c[j] / C_safe[j]
        
        # 이미 이용률이 ρ_max를 초과하는 경우: 용량 제약만 적용
        if current_utilization >= rho_max:
            constraints.append(cp.sum(x[:, j]) <= available_capacity[j] + 0.0001)
        else:
            # 용량 제약과 혼잡도 제약 중 더 엄격한 것을 적용
            capacity_limit = available_capacity[j] + 0.0001
            congestion_limit = rho_max * C_safe[j] - c[j]
            effective_limit = min(capacity_limit, max(0, congestion_limit) + 0.0001)
            constraints.append(cp.sum(x[:, j]) <= effective_limit)
        
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status not in ['optimal', 'optimal_inaccurate']:
             prob.solve(solver=cp.SCS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            return np.full(n_users, -1)
        
    # 최적화 상태 확인
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return np.full(n_users, -1)
    
    # 할당 결과 확인
    if x.value is None:
        return np.full(n_users, -1)
        
    return np.argmax(x.value, axis=1)

def calculate_metrics(assignments, users, parking_df, d_matrix):
    n_users = len(users['lat'])
    valid_mask = assignments != -1
    
    if not np.any(valid_mask):
        return {'total_distance': 0, 'avg_distance': 0, 'total_cost': 0, 'avg_cost': 0,
                'congestion_score': 0, 'unassigned': n_users, 'utilization': np.zeros(len(parking_df)),
                'counts': np.zeros(len(parking_df)), 'gini_distance': 0, 'gini_cost': 0,
                'max_distance': 0, 'max_cost': 0}
        
    valid_assignments = assignments[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    distances = d_matrix[valid_indices, valid_assignments]
    total_distance = np.sum(distances)
    
    fees = parking_df.iloc[valid_assignments]['fee_per_min'].values * 60
    times = users['time'][valid_indices]
    costs = fees * times
    total_cost = np.sum(costs)
    
    counts = np.bincount(valid_assignments, minlength=len(parking_df))
    final_counts = parking_df['current_count'].values + counts
    capacities = np.maximum(parking_df['capacity'].values, 1) 
    utilization = final_counts / capacities
    congestion_score = np.sum(utilization ** 2)
    
    return {
        'total_distance': total_distance, 'avg_distance': total_distance / np.sum(valid_mask),
        'total_cost': total_cost, 'avg_cost': total_cost / np.sum(valid_mask),
        'congestion_score': congestion_score, 'unassigned': n_users - np.sum(valid_mask),
        'utilization': utilization, 'counts': counts,
        'gini_distance': gini_coefficient(distances), 'gini_cost': gini_coefficient(costs),
        'max_distance': np.max(distances), 'max_cost': np.max(costs)
    }

def run_single_simulation(seed, parking_df, n_users, weights):
    np.random.seed(seed)
    users = generate_users(n_users, parking_df)
    d_matrix = calculate_distance_matrix(users, parking_df)
    
    start = time.time()
    random_assignments = run_random(users, parking_df, d_matrix)
    random_time = time.time() - start
    
    start = time.time()
    greedy_assignments = run_greedy(users, parking_df, d_matrix)
    greedy_time = time.time() - start
    
    start = time.time()
    convex_assignments = run_convex(users, parking_df, d_matrix, weights)
    convex_time = time.time() - start
    
    random_metrics = calculate_metrics(random_assignments, users, parking_df, d_matrix)
    greedy_metrics = calculate_metrics(greedy_assignments, users, parking_df, d_matrix)
    convex_metrics = calculate_metrics(convex_assignments, users, parking_df, d_matrix)
    
    return {
        'random': {**random_metrics, 'time': random_time},
        'greedy': {**greedy_metrics, 'time': greedy_time},
        'convex': {**convex_metrics, 'time': convex_time}
    }

def aggregate_results(all_results):
    algorithms = ['random', 'greedy', 'convex']
    metrics = ['avg_distance', 'avg_cost', 'congestion_score', 'unassigned', 
               'gini_distance', 'gini_cost', 'max_distance', 'max_cost', 'time']
    
    aggregated = {}
    for alg in algorithms:
        aggregated[alg] = {}
        for metric in metrics:
            values = [run[alg][metric] for run in all_results]
            aggregated[alg][metric] = {
                'mean': np.mean(values), 'std': np.std(values),
                'min': np.min(values), 'max': np.max(values),
                'ci_lower': np.percentile(values, 2.5), 'ci_upper': np.percentile(values, 97.5),
                'values': values
            }
    return aggregated

def statistical_tests(all_results):
    metrics = ['avg_distance', 'avg_cost', 'congestion_score']
    tests = {}
    for metric in metrics:
        greedy_vals = [run['greedy'][metric] for run in all_results]
        convex_vals = [run['convex'][metric] for run in all_results]
        t_stat, p_value = stats.ttest_rel(greedy_vals, convex_vals)
        tests[metric] = {'t_statistic': t_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    return tests

def create_visualizations(aggregated, timestamp):
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = [
        ('avg_distance', 'Average Distance (m)', 'skyblue'),
        ('congestion_score', 'Congestion Score', 'coral'),
        ('gini_distance', 'Gini Coefficient (Fairness)', 'lightgreen')
    ]
    algorithms = ['Random', 'Greedy', 'Convex']
    x_pos = np.arange(len(algorithms))
    
    for idx, (metric, title, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        means = [aggregated[alg.lower()][metric]['mean'] for alg in algorithms]
        stds = [aggregated[alg.lower()][metric]['std'] for alg in algorithms]
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=[color, color, 'purple'], alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std, f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'comparison_bars_Weekend_{timestamp}.png', dpi=150)
    print(f"Saved bar chart")
    
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (metric, title, _) in enumerate(metrics_to_plot):
        ax = axes[idx]
        data = [aggregated[alg.lower()][metric]['values'] for alg in algorithms]
        bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['skyblue', 'coral', 'purple']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_boxplots_Weekend_{timestamp}.png', dpi=150)
    print(f"Saved box plots")
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    metrics_for_improvement = ['avg_distance', 'congestion_score', 'avg_cost']
    metric_labels = ['Distance', 'Congestion', 'Cost']
    greedy_baseline = [aggregated['greedy'][m]['mean'] for m in metrics_for_improvement]
    convex_values = [aggregated['convex'][m]['mean'] for m in metrics_for_improvement]
    improvements = [(g - c) / g * 100 for g, c in zip(greedy_baseline, convex_values)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.barh(metric_labels, improvements, color=colors, alpha=0.7)
    ax.set_xlabel('Improvement over Greedy (%)', fontsize=12)
    ax.set_title('Convex Optimization Effectiveness', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    for i, (imp, bar) in enumerate(zip(improvements, bars)):
        x_pos = imp + (2 if imp > 0 else -2)
        ax.text(x_pos, i, f'{imp:+.1f}%', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'improvement_percentage_Weekend_{timestamp}.png', dpi=150)
    print(f"Saved improvement chart")

def main():
    print("="*80)
    print("Parking Allocation (NO Distance Constraint) - Weekend")
    print("="*80)
    
    data_path = os.path.join('data', 'summary_weekend_08.csv')
    n_users = 1000
    n_runs = 20
    base_weights = (0.8, 0.1, 0.1)
    
    print(f"\nLoading data from {data_path}...")
    parking_df = load_data(data_path)
    print(f"Loaded {len(parking_df)} parking lots.")
    
    total_capacity = parking_df['capacity'].sum()
    total_current = parking_df['current_count'].sum()
    print(f"Total Capacity: {total_capacity}, Current: {total_current}, Available: {total_capacity - total_current}")
    print(f"\nNO Distance Constraint - Pure Optimization")
    
    print(f"\n{'='*80}")
    print(f"Running {n_runs} independent simulations...")
    print(f"{'='*80}")
    
    all_results = []
    for i in range(n_runs):
        if (i + 1) % 5 == 0:
            print(f"Progress: {i+1}/{n_runs} runs completed")
        result = run_single_simulation(seed=42+i, parking_df=parking_df, n_users=n_users, weights=base_weights)
        all_results.append(result)
    
    aggregated = aggregate_results(all_results)
    tests = statistical_tests(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print("Creating Visualizations...")
    print(f"{'='*80}")
    create_visualizations(aggregated, timestamp)
    
    print(f"\n{'='*80}")
    print("Generating Report...")
    print(f"{'='*80}")
    
    with pd.ExcelWriter(f'weekend_report_{timestamp}.xlsx') as writer:
        summary_data = []
        for alg in ['random', 'greedy', 'convex']:
            for metric in ['avg_distance', 'avg_cost', 'congestion_score', 'gini_distance', 'time']:
                summary_data.append({
                    'Algorithm': alg.capitalize(), 'Metric': metric,
                    'Mean': aggregated[alg][metric]['mean'], 'Std': aggregated[alg][metric]['std'],
                    'CI_Lower': aggregated[alg][metric]['ci_lower'], 'CI_Upper': aggregated[alg][metric]['ci_upper']
                })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        test_data = []
        for metric, result in tests.items():
            test_data.append({
                'Metric': metric, 'T_Statistic': result['t_statistic'],
                'P_Value': result['p_value'], 'Significant_at_0.05': result['significant']
            })
        pd.DataFrame(test_data).to_excel(writer, sheet_name='Statistical_Tests', index=False)
        
        raw_data = []
        for i, run in enumerate(all_results):
            for alg in ['random', 'greedy', 'convex']:
                raw_data.append({
                    'Run': i+1, 'Algorithm': alg.capitalize(),
                    'Avg_Distance': run[alg]['avg_distance'], 'Avg_Cost': run[alg]['avg_cost'],
                    'Congestion': run[alg]['congestion_score'], 'Gini_Distance': run[alg]['gini_distance'],
                    'Gini_Cost': run[alg]['gini_cost'], 'Time_Seconds': run[alg]['time']
                })
        pd.DataFrame(raw_data).to_excel(writer, sheet_name='Raw_Results', index=False)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (NO Distance Constraint) - Weekend")
    print(f"{'='*80}\n")
    
    print("Average Distance (meters):")
    for alg in ['random', 'greedy', 'convex']:
        mean = aggregated[alg]['avg_distance']['mean']
        std = aggregated[alg]['avg_distance']['std']
        print(f"  {alg.capitalize():8s}: {mean:8.2f} ± {std:6.2f}")
    
    print("\nCongestion Score:")
    for alg in ['random', 'greedy', 'convex']:
        mean = aggregated[alg]['congestion_score']['mean']
        std = aggregated[alg]['congestion_score']['std']
        print(f"  {alg.capitalize():8s}: {mean:8.2f} ± {std:6.2f}")
    
    print("\nGini Coefficient (Distance Fairness):")
    for alg in ['random', 'greedy', 'convex']:
        mean = aggregated[alg]['gini_distance']['mean']
        std = aggregated[alg]['gini_distance']['std']
        print(f"  {alg.capitalize():8s}: {mean:8.4f} ± {std:6.4f}")
    
    print("\nComputation Time (seconds):")
    for alg in ['random', 'greedy', 'convex']:
        mean = aggregated[alg]['time']['mean']
        std = aggregated[alg]['time']['std']
        print(f"  {alg.capitalize():8s}: {mean:8.4f} ± {std:6.4f}")
    
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE (Greedy vs Convex)")
    print(f"{'='*80}\n")
    
    for metric, result in tests.items():
        sig = "YES" if result['significant'] else "NO"
        print(f"{metric:20s}: p-value = {result['p_value']:.4f} (Significant: {sig})")
    
    print(f"\n{'='*80}")
    print(f"Report saved to: weekend_report_{timestamp}.xlsx")
    print(f"{'='*80}\n")
    print("Done!")

if __name__ == "__main__":
    main()
