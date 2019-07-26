import clustering
import encoder

import numpy as np
X = np.random.random((10000, 128))

encoder = encoder.PQEncoder(num_subdim=64, Ks=256)
codewords = encoder.fit(X[:1000])
print(codewords.shape)

X_pqcode = encoder.transform(X)
print(X_pqcode.shape)

kmeans = clustering.PQKMeans(encoder=encoder, k=5)
clustered = kmeans.fit_predict(X_pqcode)
print(clustered[:100])








# import numpy
# import tqdm
# import matplotlib.pyplot as plt


# from  keras.datasets import mnist
# (img_train, _), (img_test, _) = mnist.load_data()


# print("The first image of img_train:\n")
# plt.imshow(img_train[0])


# img_trainimg_tra  = img_train[0:1000]
# img_test = img_test[0:5000]
# print("img_train.shape:\n{}".format(img_train.shape))
# print("img_test.shape:\n{}".format(img_test.shape))

# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# from scipy.misc import imresize

# base_model = VGG16(weights='imagenet')  # Read the ImageNet pre-trained VGG16 model
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)  # We use the output from the 'fc1' layer



# features_train = img_train.reshape((60000,784))
# features_test = img_test.reshape((5000,784))
# print("features_train.shape:\n{}".format(features_train.shape))
# print("features_test.shape:\n{}".format(features_test.shape))


# encoder = encoder.PQEncoder(num_subdim=4, Ks=256)
# encoder.fit(features_train)

# # Encode the deep features to PQ-codes
# pqcodes_test = encoder.transform(features_test)
# print("pqcodes_test.shape:\n{}".format(pqcodes_test.shape))
# print("pqcodes_test:\n{}".format(pqcodes_test[0]))

# # Run clustering
# K = 10
# print("Runtime of clustering:")
# clustered = clustering.PQKMeans(encoder=encoder, k=K).fit_predict(pqcodes_test)

# # for k in range(K):
# #     print("Cluster id: k={}".format(k))
# #     img_ids = [img_id for img_id, cluster_id in enumerate(clustered) if cluster_id == k]

# #     cols = 10
# #     img_ids = img_ids[0:cols] if cols < len(img_ids) else img_ids # Let's see the top 10 results
    
#     # Visualize images assigned to this cluster
#     # imgs = img_test[img_ids]
#     # plt.figure(figsize=(20, 5))
#     # for i, img in enumerate(imgs):
#     #     plt.subplot(1, cols, i + 1)
#     #     plt.imshow(img)
#     # plt.show()