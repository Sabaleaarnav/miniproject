import pandas as pd
import numpy as np
from itertools import combinations

# ML imports following hw10.py style
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

# ============================================================================
# DATA LOADING (provided starter code)
# ============================================================================

def load_data():
    """
    Loads and cleans the NYC bicycle dataset.
    Based on the provided MiniProjectPath2.py starter code.
    """
    dataset = pd.read_csv('nyc_bicycle_counts_2016.csv')
    dataset['Brooklyn Bridge'] = pd.to_numeric(dataset['Brooklyn Bridge'].replace(',', '', regex=True))
    dataset['Manhattan Bridge'] = pd.to_numeric(dataset['Manhattan Bridge'].replace(',', '', regex=True))
    dataset['Queensboro Bridge'] = pd.to_numeric(dataset['Queensboro Bridge'].replace(',', '', regex=True))
    dataset['Williamsburg Bridge'] = pd.to_numeric(dataset['Williamsburg Bridge'].replace(',', '', regex=True))
    dataset['Total'] = pd.to_numeric(dataset['Total'].replace(',', '', regex=True))
    return dataset

# ============================================================================
# QUESTION 1: BRIDGE SENSOR SELECTION
# Which 3 bridges best predict total traffic?
# ============================================================================

def get_bridge_combinations(bridges, num_select):
    """
    Returns all combinations of bridges to select.
    Following the functional style from hw4.py (compose, repeater patterns).
    
    :param bridges: list of bridge names
    :param num_select: number of bridges to select
    :return: list of tuples containing bridge combinations
    """
    return list(combinations(bridges, num_select))


def evaluate_bridge_combination(data, bridge_combo):
    """
    Evaluates how well a combination of bridges predicts total traffic.
    Uses Linear Regression following hw10.py's get_model pattern.
    
    :param data: DataFrame with bridge traffic data
    :param bridge_combo: tuple of bridge names to use as predictors
    :return: R² score, coefficients, model object
    """
    # Prepare features (X) and target (y)
    X = data[list(bridge_combo)].values
    y = data['Total'].values
    
    # Fit linear regression model (similar to hw10 model fitting pattern)
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict and calculate R² score
    y_pred = model.predict(X)
    r2_score = metrics.r2_score(y, y_pred)
    
    # Calculate RMSE for additional insight
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    
    return r2_score, rmse, model.coef_, model


def find_best_bridge_combination(data, num_bridges=3):
    """
    Finds the best combination of bridges for predicting total traffic.
    Follows the selection pattern from gmm.py's gaus_mixture (finding best k).
    
    :param data: DataFrame with bridge traffic data  
    :param num_bridges: number of bridges to select
    :return: best combination, best R², all results for analysis
    """
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    
    all_combos = get_bridge_combinations(bridges, num_bridges)
    
    # Initialize with first combination (following gmm.py pattern)
    best_combo = all_combos[0]
    best_r2, best_rmse, best_coef, best_model = evaluate_bridge_combination(data, best_combo)
    
    results = []
    
    # Iterate through all combinations to find best (like gmm.py's BIC loop)
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
    """
    Analyzes correlations between bridges to understand redundancy.
    
    :param data: DataFrame with bridge traffic data
    :return: correlation matrix
    """
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']
    return data[bridges].corr()


def question1_analysis(data):
    """
    Complete analysis for Question 1: Bridge Sensor Selection
    """
    print("=" * 70)
    print("QUESTION 1: BRIDGE SENSOR SELECTION")
    print("Which 3 bridges should sensors be installed on to best predict total?")
    print("=" * 70)
    
    # Analyze correlations first
    print("\n1.1 Correlation Analysis:")
    print("-" * 40)
    corr_matrix = analyze_bridge_correlations(data)
    print("\nCorrelation with Total Traffic:")
    for bridge in ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']:
        print(f"  {bridge}: {corr_matrix.loc[bridge, 'Total']:.4f}")
    
    # Find best 3-bridge combination
    print("\n1.2 Evaluating All 3-Bridge Combinations:")
    print("-" * 40)
    best_combo, best_r2, best_rmse, best_coef, all_results = find_best_bridge_combination(data, 3)
    
    # Sort results by R² for display
    sorted_results = sorted(all_results, key=lambda x: x['r2_score'], reverse=True)
    
    print("\nAll combinations ranked by R² score:")
    for i, result in enumerate(sorted_results, 1):
        bridges_str = ', '.join([b.replace(' Bridge', '') for b in result['bridges']])
        print(f"  {i}. [{bridges_str}]: R²={result['r2_score']:.6f}, RMSE={result['rmse']:.2f}")
    
    # Identify which bridge to exclude
    all_bridges = {'Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge'}
    excluded_bridge = (all_bridges - set(best_combo)).pop()
    
    print(f"\n1.3 RESULT:")
    print("-" * 40)
    print(f"Best 3-bridge combination: {', '.join([b.replace(' Bridge', '') for b in best_combo])}")
    print(f"Bridge to EXCLUDE: {excluded_bridge}")
    print(f"R² Score: {best_r2:.6f}")
    print(f"RMSE: {best_rmse:.2f} bicyclists")
    
    print(f"\nRegression coefficients (contribution to total):")
    for bridge, coef in zip(best_combo, best_coef):
        print(f"  {bridge}: {coef:.4f}")
    
    return best_combo, excluded_bridge, best_r2, corr_matrix


# ============================================================================
# QUESTION 2: WEATHER-BASED TRAFFIC PREDICTION
# Can weather forecast predict bicycle traffic?
# ============================================================================

def prepare_weather_features(data):
    """
    Prepares weather features for prediction.
    Following feature preparation style from gmm.py's concatenate_features.
    
    :param data: DataFrame with weather and traffic data
    :return: X (features), y (target)
    """
    # Weather features
    X = data[['High Temp', 'Low Temp', 'Precipitation']].values
    y = data['Total'].values
    return X, y


def get_weather_model(name, params):
    """
    Creates weather prediction model.
    Following hw10.py's get_model pattern exactly.
    
    :param name: Model name (string)
    :param params: list of parameters
    :return: sklearn model object
    """
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
    else:
        print("ERROR: Model name not recognized. Returned None")
    return model


def evaluate_weather_model(model_name, params, X_train, y_train, X_test, y_test):
    """
    Evaluates weather prediction model.
    Following hw10.py's get_model_results pattern.
    
    :return: r2, rmse, mae
    """
    # 1. Create model
    model = get_weather_model(model_name, params)
    
    # 2. Train the model
    model.fit(X_train, y_train)
    
    # 3. Predict on test data
    y_pred = model.predict(X_test)
    
    # 4. Calculate metrics
    r2 = metrics.r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mae = metrics.mean_absolute_error(y_test, y_pred)
    
    return r2, rmse, mae, y_pred, model


def analyze_weather_correlations(data):
    """
    Analyzes how weather variables correlate with bicycle traffic.
    """
    weather_vars = ['High Temp', 'Low Temp', 'Precipitation', 'Total']
    return data[weather_vars].corr()


def question2_analysis(data):
    """
    Complete analysis for Question 2: Weather-Based Prediction
    """
    print("\n" + "=" * 70)
    print("QUESTION 2: WEATHER-BASED TRAFFIC PREDICTION")
    print("Can next day's weather forecast predict total bicyclists?")
    print("=" * 70)
    
    # Correlation analysis
    print("\n2.1 Weather-Traffic Correlation Analysis:")
    print("-" * 40)
    weather_corr = analyze_weather_correlations(data)
    print("\nCorrelation with Total Traffic:")
    print(f"  High Temp: {weather_corr.loc['High Temp', 'Total']:.4f}")
    print(f"  Low Temp: {weather_corr.loc['Low Temp', 'Total']:.4f}")
    print(f"  Precipitation: {weather_corr.loc['Precipitation', 'Total']:.4f}")
    
    # Prepare data
    X, y = prepare_weather_features(data)
    
    # Scale features (important for MLP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (following hw10 pattern)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print("\n2.2 Model Evaluation (80-20 train-test split):")
    print("-" * 40)
    
    # Test multiple models (following hw10 pattern of testing multiple k values/models)
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
        
        param_str = str(params) if params else "default"
        print(f"\n{model_name} (params={param_str}):")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} bicyclists")
        print(f"  MAE: {mae:.2f} bicyclists")
    
    # Cross-validation for best model
    print("\n2.3 Cross-Validation Analysis (5-fold):")
    print("-" * 40)
    
    # Use Linear Regression for cross-validation
    lr_model = LinearRegression()
    cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='r2')
    print(f"Linear Regression CV R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Best model selection
    best_result = max(results, key=lambda x: x['r2'])
    
    print(f"\n2.4 RESULT:")
    print("-" * 40)
    print(f"Best Model: {best_result['model']} (R² = {best_result['r2']:.4f})")
    
    # Interpretation
    avg_traffic = data['Total'].mean()
    best_mae_pct = (best_result['mae'] / avg_traffic) * 100
    
    print(f"\nInterpretation:")
    print(f"  Average daily traffic: {avg_traffic:.0f} bicyclists")
    print(f"  Best model MAE: {best_result['mae']:.0f} bicyclists ({best_mae_pct:.1f}% of average)")
    
    if best_result['r2'] > 0.5:
        print(f"\n  CONCLUSION: Weather CAN moderately predict bicycle traffic (R² > 0.5)")
    elif best_result['r2'] > 0.3:
        print(f"\n  CONCLUSION: Weather has WEAK predictive power (0.3 < R² < 0.5)")
    else:
        print(f"\n  CONCLUSION: Weather has LIMITED predictive power (R² < 0.3)")
    
    return results, weather_corr


# ============================================================================
# QUESTION 3: DAY-OF-WEEK PATTERNS AND PREDICTION
# Can we identify weekly patterns and predict the day?
# ============================================================================

def analyze_weekly_patterns(data):
    """
    Analyzes traffic patterns by day of week.
    Computes mean and std for each day (similar to clustering centroid approach).
    
    :param data: DataFrame with traffic data
    :return: DataFrame with day-wise statistics
    """
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']
    
    # Group by day and compute statistics
    day_stats = data.groupby('Day')[bridges].agg(['mean', 'std'])
    
    return day_stats


def prepare_day_classification_data(data):
    """
    Prepares data for day-of-week classification.
    Following feature preparation style from hw10.py.
    
    :param data: DataFrame
    :return: X (features), y (labels), label_encoder
    """
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    
    X = data[bridges].values
    
    # Encode day labels
    le = LabelEncoder()
    y = le.fit_transform(data['Day'])
    
    return X, y, le


def get_day_classifier(name, params):
    """
    Creates day-of-week classifier.
    Following hw10.py's get_model pattern exactly.
    
    :param name: Model name (string)
    :param params: parameters
    :return: sklearn model object
    """
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
    else:
        print("ERROR: Model name not recognized. Returned None")
    return model


def conf_matrix(y_pred, y_true, num_class):
    """
    Creates confusion matrix manually.
    DIRECTLY from hw10.py - using the exact same implementation.
    
    :param y_pred: List of predicted classes
    :param y_true: List of corresponding true class labels
    :param num_class: The number of distinct classes being predicted
    :return: Confusion matrix as numpy array
    """
    # Initialize (from hw10.py)
    M = np.zeros((num_class, num_class))
    
    # Going through predictions and labeling (from hw10.py)
    for i in range(len(y_true)):
        correct_class = int(y_true[i])
        class_forpred = int(y_pred[i])
        M[correct_class][class_forpred] += 1
    
    return M


def evaluate_day_classifier(model_name, params, X_train, y_train, X_test, y_test, num_class):
    """
    Evaluates day-of-week classifier.
    Following hw10.py's get_model_results pattern.
    
    :return: accuracy, confusion_matrix
    """
    # 1. Create Classifier model
    model = get_day_classifier(model_name, params)
    
    # 2. Train the model using training sets
    model.fit(X_train, y_train)
    
    # 3. Predict the response for test dataset
    prediction_testdb = model.predict(X_test)
    
    # 4. Model Accuracy (from hw10.py)
    acc = metrics.accuracy_score(y_test, prediction_testdb)
    
    # 5. Calculate confusion matrix using our function (from hw10.py)
    conf_mat = conf_matrix(prediction_testdb, y_test, num_class)
    
    return acc, conf_mat, prediction_testdb, model


def question3_analysis(data):
    """
    Complete analysis for Question 3: Day-of-Week Patterns and Prediction
    """
    print("\n" + "=" * 70)
    print("QUESTION 3: DAY-OF-WEEK PATTERNS AND PREDICTION")
    print("Can we identify weekly patterns and predict the day from bridge counts?")
    print("=" * 70)
    
    # Weekly pattern analysis
    print("\n3.1 Weekly Traffic Pattern Analysis:")
    print("-" * 40)
    
    day_stats = analyze_weekly_patterns(data)
    
    # Order days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print("\nAverage Total Traffic by Day:")
    for day in day_order:
        mean_val = day_stats.loc[day, ('Total', 'mean')]
        std_val = day_stats.loc[day, ('Total', 'std')]
        print(f"  {day:10s}: {mean_val:,.0f} (+/- {std_val:,.0f})")
    
    # Identify patterns
    weekday_avg = data[data['Day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['Total'].mean()
    weekend_avg = data[data['Day'].isin(['Saturday', 'Sunday'])]['Total'].mean()
    
    print(f"\nWeekday vs Weekend:")
    print(f"  Average Weekday Traffic: {weekday_avg:,.0f}")
    print(f"  Average Weekend Traffic: {weekend_avg:,.0f}")
    print(f"  Difference: {weekday_avg - weekend_avg:,.0f} ({((weekday_avg/weekend_avg)-1)*100:.1f}% more on weekdays)")
    
    # Prepare classification data
    print("\n3.2 Day-of-Week Classification:")
    print("-" * 40)
    
    X, y, label_encoder = prepare_day_classification_data(data)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    num_class = 7  # 7 days of week
    
    # Test classifiers (following hw10.py pattern)
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
        
        param_str = str(params)
        print(f"\n{clf_name} (params={param_str}):")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    
    # Best result
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\n3.3 RESULT:")
    print("-" * 40)
    print(f"Best Classifier: {best_result['classifier']} (Accuracy = {best_result['accuracy']:.4f})")
    
    # Random baseline
    random_baseline = 1.0 / 7  # ~14.3%
    print(f"\nBaseline Comparison:")
    print(f"  Random guess baseline: {random_baseline:.4f} ({random_baseline*100:.1f}%)")
    print(f"  Best model accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.1f}%)")
    improvement = best_result['accuracy'] / random_baseline
    print(f"  Improvement over random: {improvement:.2f}x better")
    
    # Interpretation
    if best_result['accuracy'] > 0.5:
        print(f"\n  CONCLUSION: Day-of-week CAN be predicted with moderate accuracy (>{50}%)")
    elif best_result['accuracy'] > 0.3:
        print(f"\n  CONCLUSION: Day prediction is BETTER than random but challenging")
    else:
        print(f"\n  CONCLUSION: Day prediction is DIFFICULT from bridge counts alone")
    
    return day_stats, results, label_encoder


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(data, q1_results, q2_results, q3_results):
    """
    Creates visualizations for all three questions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Q1: Bridge correlations heatmap
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
    
    # Add correlation values
    for i in range(len(bridges)):
        for j in range(len(bridges)):
            ax1.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=9)
    
    # Q2: Weather vs Traffic scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(data['High Temp'], data['Total'], 
                         c=data['Precipitation'], cmap='Blues', 
                         alpha=0.6, edgecolors='gray', linewidth=0.5)
    ax2.set_xlabel('High Temperature (°F)')
    ax2.set_ylabel('Total Bicyclists')
    ax2.set_title('Q2: Temperature vs Traffic\n(color = precipitation)')
    plt.colorbar(scatter, ax=ax2, label='Precipitation (in)', shrink=0.8)
    
    # Q3: Day-of-week pattern
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
    
    # Q3: Bridge usage by day heatmap
    ax4 = axes[1, 1]
    bridges_short = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']
    day_bridge_means = np.zeros((7, 4))
    for i, day in enumerate(day_order):
        for j, bridge in enumerate(bridges):
            day_bridge_means[i, j] = data[data['Day'] == day][bridge].mean()
    
    # Normalize by column for visualization
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
    
    print("\nVisualization saved as 'bicycle_analysis_figures.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("NYC Bicycle Traffic Analysis - Path 2")
    print("=" * 70)
    print("Following coding style from: hw4.py, hw10.py, kmeans.py, gmm.py")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")
    print(f"Total records: {len(data)}")
    
    # Run analyses
    q1_results = question1_analysis(data)
    q2_results = question2_analysis(data)
    q3_results = question3_analysis(data)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(data, q1_results, q2_results, q3_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    best_combo, excluded, r2, _ = q1_results
    print(f"\nQ1: Install sensors on: {', '.join([b.replace(' Bridge', '') for b in best_combo])}")
    print(f"    Exclude: {excluded} (R² = {r2:.4f})")
    
    weather_results, _ = q2_results
    best_weather = max(weather_results, key=lambda x: x['r2'])
    print(f"\nQ2: Weather prediction R² = {best_weather['r2']:.4f}")
    print(f"    Best model: {best_weather['model']}")
    
    _, day_results, _ = q3_results
    best_day = max(day_results, key=lambda x: x['accuracy'])
    print(f"\nQ3: Day prediction accuracy = {best_day['accuracy']:.4f} ({best_day['accuracy']*100:.1f}%)")
    print(f"    Best classifier: {best_day['classifier']}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
