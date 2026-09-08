import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("hour.csv")


# --------------------------------------------------
# 2. FIRST 5 ROWS
# --------------------------------------------------

print("\nFirst 5 rows of the dataset:")
print(df.head())


# --------------------------------------------------
# 3. COLUMN METADATA
# --------------------------------------------------

descriptions = {
    'instant': 'Record index',
    'dteday': 'Date of the record',
    'season': 'Season of the year',
    'yr': 'Year indicator',
    'mnth': 'Month',
    'hr': 'Hour of the day',
    'holiday': 'Whether the day is a holiday',
    'weekday': 'Day of the week',
    'workingday': 'Whether the day is a working day',
    'weathersit': 'Weather condition',
    'temp': 'Normalized temperature',
    'atemp': 'Normalized feeling temperature',
    'hum': 'Normalized humidity',
    'windspeed': 'Normalized wind speed',
    'casual': 'Number of casual users',
    'registered': 'Number of registered users',
    'cnt': 'Total number of bike rentals'
}

metadata = pd.DataFrame({
    'Column': df.columns,
    'Description': df.columns.map(descriptions),
    'Unique Values': df.nunique(),
    'Mean': [
        df[col].mean() if pd.api.types.is_numeric_dtype(df[col])
        else np.nan
        for col in df.columns
    ],
    'Minimum': [
        df[col].min() if pd.api.types.is_numeric_dtype(df[col])
        else np.nan
        for col in df.columns
    ],
    'Maximum': [
        df[col].max() if pd.api.types.is_numeric_dtype(df[col])
        else np.nan
        for col in df.columns
    ]
})

print("\nColumn Metadata:")
print(metadata.to_string(index=False))


# --------------------------------------------------
# 4. LABELS FOR CATEGORICAL VARIABLES
# --------------------------------------------------

df['season_label'] = df['season'].map({
    1: 'Spring',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter'
})

df['year_label'] = df['yr'].map({
    0: '2011',
    1: '2012'
})

df['month_label'] = df['mnth'].map({
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
})

df['weekday_label'] = df['weekday'].map({
    0: 'Sunday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
})

df['workingday_label'] = df['workingday'].map({
    0: 'Non-working day',
    1: 'Working day'
})

df['holiday_label'] = df['holiday'].map({
    0: 'Not a holiday',
    1: 'Holiday'
})

df['weather_label'] = df['weathersit'].map({
    1: 'Clear / Partly cloudy',
    2: 'Mist / Cloudy',
    3: 'Light rain / Snow',
    4: 'Heavy rain / Snow'
})


# --------------------------------------------------
# 5. STATISTICAL SUMMARY
# --------------------------------------------------

print("\nStatistical Summary:")
print(df.describe())


# ==================================================
# MAIN EDA — 3 × 3 SUBPLOTS
# ==================================================

fig, axes = plt.subplots(
    3, 3,
    figsize=(20, 17)
)

fig.suptitle(
    "Bike Sharing Dataset - Exploratory Data Analysis",
    fontsize=20,
    fontweight='bold'
)


# --------------------------------------------------
# PLOT 1 — TARGET DISTRIBUTION
# --------------------------------------------------

sns.histplot(
    df['cnt'],
    bins=40,
    kde=True,
    ax=axes[0, 0]
)

axes[0, 0].set_title(
    "1. Distribution of Bike Rental Demand",
    fontsize=12
)

axes[0, 0].set_xlabel("Total Bike Rentals")
axes[0, 0].set_ylabel("Frequency")


# --------------------------------------------------
# PLOT 2 — HOURLY DEMAND
# --------------------------------------------------

hourly_demand = df.groupby('hr')['cnt'].mean()

axes[0, 1].plot(
    hourly_demand.index,
    hourly_demand.values,
    marker='o'
)

axes[0, 1].set_title(
    "2. Average Bike Rentals by Hour"
)

axes[0, 1].set_xlabel("Hour of Day")
axes[0, 1].set_ylabel("Average Rentals")
axes[0, 1].set_xticks(range(24))
axes[0, 1].grid(True)


# --------------------------------------------------
# PLOT 3 — SEASON VS DEMAND
# --------------------------------------------------

season_demand = df.groupby(
    'season_label'
)['cnt'].mean()

season_order = [
    'Spring',
    'Summer',
    'Fall',
    'Winter'
]

sns.barplot(
    x=season_demand.index,
    y=season_demand.values,
    order=season_order,
    ax=axes[0, 2]
)

axes[0, 2].set_title(
    "3. Average Rentals by Season"
)

axes[0, 2].set_xlabel("Season")
axes[0, 2].set_ylabel("Average Rentals")


# --------------------------------------------------
# PLOT 4 — WEATHER VS DEMAND
# --------------------------------------------------

weather_demand = df.groupby(
    'weather_label'
)['cnt'].mean()

weather_order = [
    'Clear / Partly cloudy',
    'Mist / Cloudy',
    'Light rain / Snow',
    'Heavy rain / Snow'
]

sns.barplot(
    x=weather_demand.index,
    y=weather_demand.values,
    order=weather_order,
    ax=axes[1, 0]
)

axes[1, 0].set_title(
    "4. Average Rentals by Weather"
)

axes[1, 0].set_xlabel("Weather Condition")
axes[1, 0].set_ylabel("Average Rentals")

axes[1, 0].tick_params(
    axis='x',
    rotation=25
)


# --------------------------------------------------
# PLOT 5 — WORKING DAY VS DEMAND
# --------------------------------------------------

workingday_demand = df.groupby(
    'workingday_label'
)['cnt'].mean()

sns.barplot(
    x=workingday_demand.index,
    y=workingday_demand.values,
    order=[
        'Non-working day',
        'Working day'
    ],
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "5. Working Day vs Non-working Day"
)

axes[1, 1].set_xlabel("Day Type")
axes[1, 1].set_ylabel("Average Rentals")


# --------------------------------------------------
# PLOT 6 — TEMPERATURE VS DEMAND
# LINEAR TREND
# --------------------------------------------------

sns.scatterplot(
    data=df,
    x='temp',
    y='cnt',
    alpha=0.25,
    ax=axes[1, 2]
)

sns.regplot(
    data=df,
    x='temp',
    y='cnt',
    scatter=False,
    ax=axes[1, 2]
)

axes[1, 2].set_title(
    "6. Temperature vs Rental Demand"
)

axes[1, 2].set_xlabel(
    "Normalized Temperature"
)

axes[1, 2].set_ylabel(
    "Bike Rentals"
)


# --------------------------------------------------
# PLOT 7 — HOUR × WORKING DAY × DEMAND
# MULTI-PARAMETER
# --------------------------------------------------

sns.lineplot(
    data=df,
    x='hr',
    y='cnt',
    hue='workingday_label',
    estimator='mean',
    ax=axes[2, 0]
)

axes[2, 0].set_title(
    "7. Hour × Working Day × Rental Demand"
)

axes[2, 0].set_xlabel("Hour of Day")
axes[2, 0].set_ylabel("Average Rentals")
axes[2, 0].set_xticks(range(24))
axes[2, 0].grid(True)


# --------------------------------------------------
# PLOT 8 — TEMPERATURE × HUMIDITY × DEMAND
# MULTI-PARAMETER
# --------------------------------------------------

scatter = axes[2, 1].scatter(
    df['temp'],
    df['cnt'],
    c=df['hum'],
    alpha=0.5
)

axes[2, 1].set_title(
    "8. Temperature × Humidity × Rental Demand"
)

axes[2, 1].set_xlabel(
    "Normalized Temperature"
)

axes[2, 1].set_ylabel(
    "Bike Rentals"
)

fig.colorbar(
    scatter,
    ax=axes[2, 1],
    label="Normalized Humidity"
)


# --------------------------------------------------
# PLOT 9 — CORRELATION MATRIX
# --------------------------------------------------

numeric_columns = [
    'temp',
    'atemp',
    'hum',
    'windspeed',
    'hr',
    'season',
    'mnth',
    'weekday',
    'workingday',
    'cnt'
]

correlation = df[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=axes[2, 2]
)

axes[2, 2].set_title(
    "9. Correlation Matrix"
)


# --------------------------------------------------
# FINAL SPACING
# --------------------------------------------------

plt.tight_layout(
    rect=[0, 0, 1, 0.95],
    pad=3.0,
    h_pad=3.0,
    w_pad=3.0
)

plt.show()


# ==================================================
# OUTLIER DETECTION — 2 × 2
# ==================================================

outlier_features = [
    'temp',
    'atemp',
    'hum',
    'windspeed'
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10)
)

fig.suptitle(
    "Outlier Detection using Boxplots",
    fontsize=18,
    fontweight='bold'
)


for i, feature in enumerate(outlier_features):

    row = i // 2
    col = i % 2

    sns.boxplot(
        x=df[feature],
        ax=axes[row, col]
    )

    axes[row, col].set_title(
        f"{feature} - Outlier Detection",
        fontsize=12
    )

    axes[row, col].set_xlabel(feature)


# --------------------------------------------------
# SPACING FOR OUTLIER PLOTS
# --------------------------------------------------

plt.tight_layout(
    rect=[0, 0, 1, 0.93],
    pad=3.0,
    h_pad=3.0,
    w_pad=3.0
)

plt.show()


# ==================================================
# NUMERICAL OUTLIER ANALYSIS
# ==================================================

print("\nOutlier Analysis:")

for feature in outlier_features:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[feature] < lower_limit) |
        (df[feature] > upper_limit)
    ]

    percentage = (
        len(outliers) / len(df)
    ) * 100

    print(
        f"{feature}: "
        f"{len(outliers)} outliers "
        f"({percentage:.2f}%)"
    )


# ==================================================
# TARGET CORRELATION
# ==================================================

target_correlation = correlation[
    'cnt'
].sort_values(
    ascending=False
)

print(
    "\nCorrelation of Features with Bike Rental Demand:"
)

print(target_correlation)


print("\nEDA completed successfully.")