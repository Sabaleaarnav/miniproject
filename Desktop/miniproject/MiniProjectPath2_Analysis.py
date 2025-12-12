import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
    print("=" * 60)
    print("QUESTION 1: BRIDGE SENSOR SELECTION")
    print("=" * 60)
    
    corr_matrix = analyze_bridge_correlations(data)
    print("\nCorrelation with Total Traffic:")
    for bridge in ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']:
        print(f"  {bridge}: {corr_matrix.loc[bridge, 'Total']:.4f}")
    
    best_combo, best_r2, best_rmse, best_coef, all_results = find_best_bridge_combination(data, 3)
    
    sorted_results = sorted(all_results, key=lambda x: x['r2_score'], reverse=True)
    
    print("\nAll 3-bridge combinations ranked by R² score:")
    for i, result in enumerate(sorted_results, 1):
        bridges_str = ', '.join([b.replace(' Bridge', '') for b in result['bridges']])
        print(f"  {i}. [{bridges_str}]: R²={result['r2_score']:.6f}, RMSE={result['rmse']:.2f}")
    
    all_bridges = {'Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'}
    excluded_bridge = (all_bridges - set(best_combo)).pop()
    
    print(f"\nBest combination: {', '.join([b.replace(' Bridge', '') for b in best_combo])}")
    print(f"Bridge to EXCLUDE: {excluded_bridge}")
    print(f"R² Score: {best_r2:.6f}")
    print(f"RMSE: {best_rmse:.2f} bicyclists")
    
    return best_combo, excluded_bridge, best_r2, corr_matrix

def prepare_weather_features(data):
    X = data[['High Temp', 'Low Temp', 'Precipitation']].values
    y = data['Total'].values
    return X, y

def get_weather_model(name, params):
    model = None
    if name == "Linear":
        model = LinearRegression()
    elif name == "KNN_Reg":
        k = params
        model = KNeighborsRegressor(n_neighbors=k)
    elif name == "MLP_Reg":
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
    print("\n" + "=" * 60)
    print("QUESTION 2: WEATHER-BASED TRAFFIC PREDICTION")
    print("=" * 60)
    
    weather_corr = analyze_weather_correlations(data)
    print("\nCorrelation with Total Traffic:")
    print(f"  High Temp: {weather_corr.loc['High Temp', 'Total']:.4f}")
    print(f"  Low Temp: {weather_corr.loc['Low Temp', 'Total']:.4f}")
    print(f"  Precipitation: {weather_corr.loc['Precipitation', 'Total']:.4f}")
    
    X, y = prepare_weather_features(data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    models_to_test = [
        ("Linear", None),
        ("KNN_Reg", 5),
        ("KNN_Reg", 10),
        ("MLP_Reg", [(50, 25), 42, "relu"]),
    ]
    
    print("\nModel Performance (80-20 train-test split):")
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
            'mae': mae
        })
        
        param_str = str(params) if params else "default"
        print(f"\n  {model_name} (params={param_str}):")
        print(f"    R² Score: {r2:.4f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE: {mae:.2f}")
    
    best_result = max(results, key=lambda x: x['r2'])
    avg_traffic = data['Total'].mean()
    
    print(f"\nBest Model: {best_result['model']} with R² = {best_result['r2']:.4f}")
    print(f"Average daily traffic: {avg_traffic:.0f}")
    print(f"Best MAE as % of average: {(best_result['mae']/avg_traffic)*100:.1f}%")
    
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
    elif name == "SVM":
        rand_state, prob = params
        model = SVC(random_state=rand_state, probability=prob)
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
    print("\n" + "=" * 60)
    print("QUESTION 3: DAY-OF-WEEK PATTERNS AND PREDICTION")
    print("=" * 60)
    
    day_stats = analyze_weekly_patterns(data)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print("\nAverage Total Traffic by Day:")
    for day in day_order:
        mean_val = day_stats.loc[day, ('Total', 'mean')]
        std_val = day_stats.loc[day, ('Total', 'std')]
        print(f"  {day:10s}: {mean_val:,.0f} (+/- {std_val:,.0f})")
    
    weekday_avg = data[data['Day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['Total'].mean()
    weekend_avg = data[data['Day'].isin(['Saturday', 'Sunday'])]['Total'].mean()
    
    print(f"\nWeekday average: {weekday_avg:,.0f}")
    print(f"Weekend average: {weekend_avg:,.0f}")
    print(f"Weekday has {((weekday_avg/weekend_avg)-1)*100:.1f}% more traffic than weekend")
    
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
        ("SVM", [42, True]),
        ("MLP", [(64, 32), 42, "relu"]),
    ]
    
    print("\nDay-of-Week Classification Results:")
    results = []
    for clf_name, params in classifiers:
        acc, conf_mat, y_pred, model = evaluate_day_classifier(
            clf_name, params, X_train, y_train, X_test, y_test, num_class
        )
        results.append({
            'classifier': clf_name,
            'params': params,
            'accuracy': acc,
            'confusion_matrix': conf_mat
        })
        
        print(f"\n  {clf_name} (params={params}):")
        print(f"    Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    
    best_result = max(results, key=lambda x: x['accuracy'])
    random_baseline = 1.0 / 7
    
    print(f"\nBest Classifier: {best_result['classifier']}")
    print(f"Best Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.1f}%)")
    print(f"Random baseline: {random_baseline:.4f} ({random_baseline*100:.1f}%)")
    print(f"Improvement over random: {best_result['accuracy']/random_baseline:.2f}x")
    
    return day_stats, results, label_encoder

if __name__ == "__main__":
    data = load_data()
    print(f"Dataset loaded: {len(data)} records")
    print(f"Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")
    
    q1_results = question1_analysis(data)
    q2_results = question2_analysis(data)
    q3_results = question3_analysis(data)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_combo, excluded, r2, _ = q1_results
    print(f"\nQ1: Install sensors on {', '.join([b.replace(' Bridge', '') for b in best_combo])}")
    print(f"    Exclude: {excluded} (R² = {r2:.4f})")
    
    weather_results, _ = q2_results
    best_weather = max(weather_results, key=lambda x: x['r2'])
    print(f"\nQ2: Best weather model: {best_weather['model']} (R² = {best_weather['r2']:.4f})")
    
    _, day_results, _ = q3_results
    best_day = max(day_results, key=lambda x: x['accuracy'])
    print(f"\nQ3: Best day classifier: {best_day['classifier']} (Accuracy = {best_day['accuracy']:.4f})")