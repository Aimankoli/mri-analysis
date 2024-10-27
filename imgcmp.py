import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt


def svd(m: np.matrix):

    rank=np.linalg.matrix_rank(m)
    
    mm_t = np.dot(m, m.T)  
    m_tm = np.dot(m.T, m)  

    eigvals_u, u = np.linalg.eig(mm_t)

    eigvals_v, v = np.linalg.eig(m_tm)

    idx_u = np.argsort(eigvals_u)[::-1]  # Sort in descending order
    idx_v = np.argsort(eigvals_v)[::-1]  # Sort in descending order

    u = u[:, idx_u]
    v = v[:, idx_v]

    eigvals_u = eigvals_u[idx_u]
    eigvals_u = eigvals_u[:rank]
    
    u = u[:, :rank] 
    v = v[:, :rank]


    # Step 6: The singular values are the square roots of the sorted eigenvalues
    singular_values = np.sqrt(np.abs(eigvals_u))
    sigma_matrix = np.diag(singular_values)

    # Step 7: Normalize U and V to make sure they are orthonormal
    u = u / np.linalg.norm(u, axis=0)
    v = v / np.linalg.norm(v, axis=0)

    
    # Step 8: Transpose V to get V^T
    v_t = v.T

    # Return U, Sigma, and V^T
    return u, sigma_matrix, v_t

import os
def dir():
    print("Current Working Directory: ", os.getcwd())

def preprocess(img: str):
    image = im.open(img)
    A = np.array(image, dtype=float) / 255.0
    A = np.dot(A, [0.2989, 0.5870, 0.1140])
    plt.imshow(A)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    return A

def compress(rank: int, image: np.array):


    u,s,v = np.linalg.svd(image, full_matrices=False)
    s = np.diag(s)
    a = np.dot(u, np.dot(s,v))
    a = np.real(a)

   

    udash = u[:, :rank]
    sdash = s[:rank, :rank]
    vdash = v[:rank, :]
    compressed_image = np.dot(udash, np.dot(sdash, vdash))
    compressed_image = np.real(compressed_image)
    plt.imshow(compressed_image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

    return compressed_image

def fnorm(matrix: np.array, img: np.array):
    # Calculate the difference between the matrices
    diff = matrix - img
    
    # Compute the Frobenius norm using numpy operations
    norm = np.sqrt(np.sum(np.square(diff)))
    
    return norm

def main():
    A = preprocess('tj.jpg')
    
    ranks =[300]
    print(A)
    fnorms=[]
    for r in ranks:
        img = compress(r, A)
        fnorms.append(fnorm(A, img))
    print (f"Fnorms: ", fnorms)


if __name__ =="__main__":
    main()





