from sklearn.datasets import load_iris

iris = load_iris()

print("Nombre d'échantillons :", iris.data.shape[0])
print("Nombre de variables :", iris.data.shape[1])
print("Classes :", iris.target_names)


