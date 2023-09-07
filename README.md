# CBSVD

In this project we recreated a new approach to improve the accuracy and scalability of two factorized collaborative filtering models: SVD++ and Asymmetric-SVD++ suggested by the paper “Clustering-Based Factorized Collaborative Filtering” by Nima Mirbakhsh and Charles X. Ling. 

The implementation is done using Python (skikit-surprise library) and can be divided into three main steps: Calculating the latent vectors of the users and items, calculating the k-Means on the latent vectors to create a mean latent vector for each category, and finally adding the mean latent vectors to the prediction algorithm.

Contributors: Alan Kachamzov, Mario Jani Dangev, Julian Drexler, Szilard Gili
