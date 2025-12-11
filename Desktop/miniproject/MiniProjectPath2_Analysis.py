import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_data():
    dataset = pd.read_csv('nyc_bicycle_counts_2016.csv')
    dataset['Brooklyn Bridge'] = pd.to_numeric(dataset['Brooklyn Bridge'].replace(',', '', regex=True))
    dataset['Manhattan Bridge'] = pd.to_numeric(dataset['Manhattan Bridge'].replace(',', '', regex=True))
    dataset['Queensboro Bridge'] = pd.to_numeric(dataset['Queensboro Bridge'].replace(',', '', regex=True))
    dataset['Williamsburg Bridge'] = pd.to_numeric(dataset['Williamsburg Bridge'].replace(',', '', regex=True))
    dataset['Total'] = pd.to_numeric(dataset['Total'].replace(',', '', regex=True))
    return dataset

def get_bridge_combinations(bridges, num_select):
    return list(combinations(bridges, num_select))

def evaluate_bridge_combination(data, bridge_combo):
    X = data[list(bridge_combo)].values
    y = data['Total'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2_score = metrics.r2_score(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    
    return r2_score, rmse, model.coef_, model

def find_best_bridge_combination(data, num_bridges=3):
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    all_combos = get_bridge_combinations(bridges, num_bridges)
    
    best_combo = all_combos[0]
    best_r2, best_rmse, best_coef, best_model = evaluate_bridge_combination(data, best_combo)
    
    results = []
    
    for combo in all_combos:
        r2, rmse, coef, model = evaluate_bridge_combination(data, combo)
        results.append({
            'bridges': combo,
            'r2_score': r2,
            'rmse': rmse,
            'coefficients': coef
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_rmse = rmse
            best_combo = combo
            best_coef = coef
            best_model = model
    
    return best_combo, best_r2, best_rmse, best_coef, results

def analyze_bridge_correlations(data):
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']
    return data[bridges].corr()

def question1_analysis(data):
    corr_matrix = analyze_bridge_correlations(data)
    best_combo, best_r2, best_rmse, best_coef, all_results = find_best_bridge_combination(data, 3)
    
    all_bridges = {'Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'}
    excluded_bridge = (all_bridges - set(best_combo)).pop()
    
    return best_combo, excluded_bridge, best_r2, corr_matrix

def prepare_weather_features(data):
    X = data[['High Temp', 'Low Temp', 'Precipitation']].values
    y = data['Total'].values
    return X, y

def get_weather_model(name, params):
    model = None
    if name == "Linear":
        model = LinearRegression()
    elif name == "Ridge":
        alpha = params
        model = Ridge(alpha=alpha)
    elif name == "KNN_Reg":
        from sklearn.neighbors import KNeighborsRegressor
        k = params
        model = KNeighborsRegressor(n_neighbors=k)
    elif name == "MLP_Reg":
        from sklearn.neural_network import MLPRegressor
        hl_sizes, rand_state, act_func = params
        model = MLPRegressor(hidden_layer_sizes=hl_sizes, random_state=rand_state, 
                            activation=act_func, max_iter=1000)
    return model

def evaluate_weather_model(model_name, params, X_train, y_train, X_test, y_test):
    model = get_weather_model(model_name, params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = metrics.r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mae = metrics.mean_absolute_error(y_test, y_pred)
    
    return r2, rmse, mae, y_pred, model

def analyze_weather_correlations(data):
    weather_vars = ['High Temp', 'Low Temp', 'Precipitation', 'Total']
    return data[weather_vars].corr()

def question2_analysis(data):
    weather_corr = analyze_weather_correlations(data)
    X, y = prepare_weather_features(data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    models_to_test = [
        ("Linear", None),
        ("Ridge", 1.0),
        ("KNN_Reg", 5),
        ("KNN_Reg", 10),
        ("MLP_Reg", [(50, 25), 42, "relu"]),
    ]
    
    results = []
    for model_name, params in models_to_test:
        r2, rmse, mae, y_pred, model = evaluate_weather_model(
            model_name, params, X_train, y_train, X_test, y_test
        )
        results.append({
            'model': model_name,
            'params': params,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred,
            'model_obj': model
        })
    
    return results, weather_corr

def analyze_weekly_patterns(data):
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']
    day_stats = data.groupby('Day')[bridges].agg(['mean', 'std'])
    return day_stats

def prepare_day_classification_data(data):
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    X = data[bridges].values
    
    le = LabelEncoder()
    y = le.fit_transform(data['Day'])
    
    return X, y, le

def get_day_classifier(name, params):
    model = None
    if name == "KNN":
        k = params
        model = KNeighborsClassifier(n_neighbors=k)
    elif name == "RandomForest":
        n_est, rand_state = params
        model = RandomForestClassifier(n_estimators=n_est, random_state=rand_state)
    elif name == "MLP":
        hl_sizes, rand_state, act_func = params
        model = MLPClassifier(hidden_layer_sizes=hl_sizes, random_state=rand_state, 
                             activation=act_func, max_iter=1000)
    return model

def conf_matrix(y_pred, y_true, num_class):
    M = np.zeros((num_class, num_class))
    for i in range(len(y_true)):
        correct_class = int(y_true[i])
        class_forpred = int(y_pred[i])
        M[correct_class][class_forpred] += 1
    return M

def evaluate_day_classifier(model_name, params, X_train, y_train, X_test, y_test, num_class):
    model = get_day_classifier(model_name, params)
    model.fit(X_train, y_train)
    prediction_testdb = model.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, prediction_testdb)
    conf_mat = conf_matrix(prediction_testdb, y_test, num_class)
    
    return acc, conf_mat, prediction_testdb, model

def question3_analysis(data):
    day_stats = analyze_weekly_patterns(data)
    
    X, y, label_encoder = prepare_day_classification_data(data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    num_class = 7
    
    classifiers = [
        ("KNN", 3),
        ("KNN", 5),
        ("RandomForest", [100, 42]),
        ("MLP", [(64, 32), 42, "relu"]),
    ]
    
    results = []
    for clf_name, params in classifiers:
        acc, conf_mat, y_pred, model = evaluate_day_classifier(
            clf_name, params, X_train, y_train, X_test, y_test, num_class
        )
        results.append({
            'classifier': clf_name,
            'params': params,
            'accuracy': acc,
            'confusion_matrix': conf_mat,
            'model': model
        })
    
    return day_stats, results, label_encoder

def create_visualizations(data, q1_results, q2_results, q3_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    bridge_short = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']
    corr_data = data[bridges].corr()
    im1 = ax1.imshow(corr_data, cmap='YlOrRd', aspect='auto', vmin=0.7, vmax=1)
    ax1.set_xticks(range(len(bridge_short)))
    ax1.set_yticks(range(len(bridge_short)))
    ax1.set_xticklabels(bridge_short, rotation=45, ha='right')
    ax1.set_yticklabels(bridge_short)
    ax1.set_title('Q1: Bridge Traffic Correlations')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    for i in range(len(bridges)):
        for j in range(len(bridges)):
            ax1.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=9)
    
    ax2 = axes[0, 1]
    scatter = ax2.scatter(data['High Temp'], data['Total'], 
                         c=data['Precipitation'], cmap='Blues', 
                         alpha=0.6, edgecolors='gray', linewidth=0.5)
    ax2.set_xlabel('High Temperature (°F)')
    ax2.set_ylabel('Total Bicyclists')
    ax2.set_title('Q2: Temperature vs Traffic\n(color = precipitation)')
    plt.colorbar(scatter, ax=ax2, label='Precipitation (in)', shrink=0.8)
    
    ax3 = axes[1, 0]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_means = [data[data['Day'] == day]['Total'].mean() for day in day_order]
    day_stds = [data[data['Day'] == day]['Total'].std() for day in day_order]
    
    colors = ['#2ecc71' if day not in ['Saturday', 'Sunday'] else '#e74c3c' for day in day_order]
    bars = ax3.bar(range(7), day_means, yerr=day_stds, capsize=5, color=colors, alpha=0.7)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax3.set_ylabel('Average Total Bicyclists')
    ax3.set_title('Q3: Weekly Traffic Pattern\n(Green=Weekday, Red=Weekend)')
    ax3.axhline(y=np.mean(day_means), color='black', linestyle='--', alpha=0.5, label='Overall Mean')
    ax3.legend()
    
    ax4 = axes[1, 1]
    bridges_short = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']
    day_bridge_means = np.zeros((7, 4))
    for i, day in enumerate(day_order):
        for j, bridge in enumerate(bridges):
            day_bridge_means[i, j] = data[data['Day'] == day][bridge].mean()
    
    day_bridge_norm = day_bridge_means / day_bridge_means.max(axis=0)
    
    im4 = ax4.imshow(day_bridge_norm, cmap='YlGn', aspect='auto')
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(7))
    ax4.set_xticklabels(bridges_short, rotation=45, ha='right')
    ax4.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax4.set_title('Q3: Normalized Bridge Usage by Day')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='Relative Usage')
    
    plt.tight_layout()
    plt.savefig('bicycle_analysis_figures.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data = load_data()
    
    q1_results = question1_analysis(data)
    q2_results = question2_analysis(data)
    q3_results = question3_analysis(data)
    
    create_visualizations(data, q1_results, q2_results, q3_results)