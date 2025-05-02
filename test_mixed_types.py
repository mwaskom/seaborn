import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a test DataFrame with mixed data types
np.random.seed(42)
n = 100

# Create a column with mixed types (int and str)
mixed_data = []
for i in range(n):
    if i % 5 == 0:
        mixed_data.append(f"SGAFC{i}DP")  # String value
    else:
        mixed_data.append(i % 10)  # Integer value

# Create DataFrame
df = pd.DataFrame({
    'mixed_col': mixed_data,
    'category': np.random.choice(['A', 'B', 'C'], n)
})

# Verify the column has object dtype with mixed types
print("Column dtype:", df['mixed_col'].dtype)
print("Sample values:", df['mixed_col'].head(10).tolist())

# Test case 1: Basic histogram with mixed data types - should work with our fix
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
sns.histplot(df, x='mixed_col')
plt.title("Basic histogram with mixed types")

# Test case 2: With hue parameter - would previously fail
plt.subplot(2, 2, 2)
sns.histplot(df, x='mixed_col', hue='category')
plt.title("Histogram with hue (would previously fail)")

# Test case 3: Manually convert to string for comparison
plt.subplot(2, 2, 3)
sns.histplot(df.astype(str), x='mixed_col', hue='category')
plt.title("Manual string conversion (previously worked)")

# Test case 4: Original workaround
plt.subplot(2, 2, 4)
df_copy = df.copy()
df_copy['mixed_col'] = df_copy['mixed_col'].astype(str)
sns.histplot(df_copy, x='mixed_col', hue='category')
plt.title("Original workaround")

plt.tight_layout()
plt.savefig('histplot_mixed_types_test.png')
print("Test script completed successfully!") 