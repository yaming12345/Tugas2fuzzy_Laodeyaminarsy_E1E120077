import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fuzzy_c_means(X, c, m, max_iters, error_threshold):
    n, features = X.shape
    
    # Inisialisasi pusat kelompok dan matriks keanggotaan
    centers = np.random.rand(c, features)
    U = np.random.rand(n, c)
    U = U / np.sum(U, axis=1, keepdims=True)  # Normalisasi keanggotaan
    
    for _ in range(max_iters):
        # Perhitungan tingkat keanggotaan
        distances = np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2)
        U_new = 1 / np.power(distances, 2/(m-1))
        U_new /= np.sum(U_new, axis=1, keepdims=True)  # Normalisasi keanggotaan

        # Perhitungan pusat kelompok baru
        centers = np.dot(U_new.T, X) / np.sum(U_new, axis=0, keepdims=True).T

        # Hitung error atau konvergensi
        error = np.sum(np.abs(U_new - U))
        
        # Update matriks keanggotaan
        U = U_new

        # Cek konvergensi
        if error < error_threshold:
            break
    
    return centers, U

# File CSV dan kolom
file_path = "video_games_sales.csv"
selected_columns = ["NA_Sales", "EU_Sales"]  # Ganti dengan nama kolom yang sesuai pada file CSV

# Baca data dari file CSV
data = pd.read_csv(file_path)

# Pilih kolom yang digunakan
X = data[selected_columns].values

# Jumlah kelompok
c = 3  # Ganti sesuai kebutuhan

# Parameter fuzziness
m = 2

# Jumlah iterasi maksimum dan threshold error
max_iters = 100
error_threshold = 1e-5

# Panggil fungsi FCM
centers, U = fuzzy_c_means(X, c, m, max_iters, error_threshold)

# Menampilkan output dalam bentuk tabel
output_df = pd.DataFrame(U, columns=[f'Cluster {i+1}' for i in range(c)])
output_df.insert(0, 'Data', range(1, len(U)+1))
output_df.set_index('Data', inplace=True)

print("\nPusat Kelompok:")
print(pd.DataFrame(centers, columns=selected_columns))
print("\nMatriks Keanggotaan:")
print(output_df)

# Menampilkan scatter plot
plt.figure(figsize=(10, 6))
for i in range(c):
    plt.scatter(X[:, 0], X[:, 1], c=U[:, i], cmap='viridis', s=50, edgecolors='k', label=f'Cluster {i+1}', alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.legend()
plt.show()
