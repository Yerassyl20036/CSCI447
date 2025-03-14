import numpy as np
import matplotlib.pyplot as plt

#linear mapping
X = np.random.randn(1, 100) + 1  # Mean 1 
Y = np.random.randn(1, 100) + 2  # Mean 2 
data = np.vstack((X, Y))  # Concatinate

A = np.array([[0.5, 0],  
              [0, 1]])   
transformed_data_A = A @ data 

B = np.array([[-1, 0],  
              [0, 1]])   
transformed_data_B = B @ transformed_data_A  

C = np.array([[0, 1],
              [1, 0]])   
transformed_data_C = C @ transformed_data_B  

theta = np.radians(45)
D = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])  
transformed_data_D = D @ transformed_data_C  

E = D @ C @ B @ A
transformed_data_E = E @ data  

F = A @ B @ C @ D
transformed_data_F = F @ data  


#Plot
plt.figure(figsize=(8, 8))

plt.plot(data[0, :], data[1, :], '.', markersize=8, label="Data", alpha=0.7)
plt.plot(transformed_data_A[0, :], transformed_data_A[1, :], '.', markersize=8, label="A", alpha=0.7)
plt.plot(transformed_data_B[0, :], transformed_data_B[1, :], '.', markersize=8, label="B", alpha=0.7)
plt.plot(transformed_data_C[0, :], transformed_data_C[1, :], '.', markersize=8, label="C", alpha=0.7)
plt.plot(transformed_data_D[0, :], transformed_data_D[1, :], '.', markersize=8, label="D", alpha=0.7)
plt.plot(transformed_data_E[0, :], transformed_data_E[1, :], '.', markersize=8, label="E", alpha=0.7)
plt.plot(transformed_data_F[0, :], transformed_data_F[1, :], '.', markersize=8, label="F", alpha=0.7)

plt.title('Linear Mappings')
plt.legend()
plt.grid(True)
plt.show()
