import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import untuk plotting 3D

# 1. Load dataset jenis CSV
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File CSV '{file_path}' tidak ditemukan. Program berhenti.")
        exit()
    except pd.errors.EmptyDataError:
        print(f"File CSV '{file_path}' tidak berisi data. Program berhenti.")
        exit()
    except Exception as e:
        print(f"Terjadi kesalahan: {e}. Program berhenti.")
        exit()

# 2. Pilih kolom yang akan dijadikan fitur
def choose_features(data, selected_columns):
    # Memastikan nama kolom yang dimasukkan benar
    if all(feature in data.columns for feature in selected_columns):
        return data[selected_columns]
    else:
        print("Salah satu atau beberapa nama kolom yang dimasukkan tidak sesuai. Program berhenti.")
        exit()

# 3. Run data
def run_kmeans(data, cluster_count):
    # Mengatasi nilai NaN dengan mengisi nilai rata-rata dari setiap kolom
    data = data.fillna(data.mean())

    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_

    # Tampilkan informasi nilai centroid
    print("\nNilai Centroid:")
    for i in range(cluster_count):
        print(f"Cluster {i + 1}: {centroids[i]}")

    # Visualisasi hasil clustering
    if data.shape[1] == 2:
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='rainbow')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black')
        plt.title('Hasil K-means Clustering')
        plt.show()
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=labels, cmap='rainbow')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='X', s=200, c='black')
        ax.set_title('Hasil K-means Clustering (3D)')
        plt.show()
    else:
        print("Visualisasi hanya mendukung data dengan 2 atau 3 dimensi.")

if __name__ == "__main__":
    # Tetapkan file CSV dan kolom yang akan di clustering
    file_path = "video_games_sales.csv"
    selected_columns = ["NA_Sales", "EU_Sales"]  # Ganti dengan nama kolom yang sesuai

    # 1. Load dataset jenis CSV
    dataset = load_dataset(file_path)

    # 2. Pilih kolom yang akan dijadikan fitur
    selected_features = choose_features(dataset, selected_columns)

    # 3. Pilih jumlah cluster
    num_clusters = int(input("Masukkan jumlah cluster: "))

    # 4. Run data
    run_kmeans(selected_features, num_clusters)
