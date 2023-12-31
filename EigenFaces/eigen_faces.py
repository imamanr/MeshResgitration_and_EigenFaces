from PIL import Image
import numpy as np
import os


images = np.empty((68*64, 120),dtype=np.float32)
size = 68, 64
for i,f in enumerate(os.listdir("./shape_normalized_images")):
    path = os.path.join("./shape_normalized_images",f)
    im = np.asarray(Image.open(path).resize(size).convert('L'))
    images[:, i] = im.reshape(-1)/255

mean = np.mean(images, axis=1)
mean_img = Image.fromarray(255*mean.reshape(64, 68)).convert('L')
mean_img.save("results/mean.jpg")
images = images - mean.reshape(-1,1)
cov = np.dot(images.T, images)

eigen_value, eigen_vector = np.linalg.eig(cov)
idx = eigen_value.argsort()[::-1]
eigenvalues = eigen_value[idx]
eigenvectors = eigen_vector[:, idx]
eigen_faces = np.dot(images, eigenvectors)
file_name = "results/eigenface_"
if not os.path.exists("results"):
    os.mkdir("results")
for i in range(10):
    f = file_name + str(i)+".jpg"
    m = 255*eigen_faces[:, i].reshape(64, 68)
    im = Image.fromarray(m).convert('L')
    im.save(f)




