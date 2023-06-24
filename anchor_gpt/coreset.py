###
# Implementation of the Coreset Algirithm
# Heavily inspired by https://github.com/google/active-learning
###


import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random

import tempfile

random.seed(1234)
np.random.seed(1234)

try:
    # Using cuML to GPU enable coreset, otherwise use sklearn
    from cuml.metrics.pairwise_distances import pairwise_distances
except Exception as e:
    from sklearn.metrics import pairwise_distances

BATCH_SIZE = int(os.environ.get('CORESET_BATCH_SIZE', '100'))

class kCenterGreedy():

  def __init__(self, X, cache_dir, vector_shape, batch_shapes, metric='euclidean'):
    self.X = X
    self.metric = metric
    self.min_distances = None
    self.already_selected = []
    self.cache_dir = cache_dir
    self.vector_shape = vector_shape
    self.batch_shapes = batch_shapes

  def get_vectors(self, uuids):
    vector_array = np.empty(shape=(len(uuids), *self.vector_shape), dtype='float16')
    for index, uuid in enumerate(uuids):
      vector_array[index] = np.memmap(
        os.path.join(self.cache_dir, str(uuid)), dtype='float16', mode='r', shape=self.vector_shape)
    return vector_array

  def get_batches(self, batch_start):
    return np.memmap(
        os.path.join(self.cache_dir, str(batch_start)), dtype='float16', mode='r',
        shape=self.batch_shapes[batch_start])


  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
    cluster_centers = np.array(cluster_centers)
    if len(cluster_centers):
      dist_list = []
      x = self.get_vectors(self.X[cluster_centers])

      for batch_start in range(0, len(self.X), BATCH_SIZE):
        features_batch = self.get_batches(batch_start)
        batch_dist = pairwise_distances(features_batch, x, metric=self.metric)
        dist_list.append(batch_dist)
      dist = np.concatenate(dist_list)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, already_selected, N):

    self.update_distances(already_selected, only_new=False, reset_dist=True)

    new_batch = []

    n_obs = len(self.X)

    for _ in tqdm(range(N), desc='calculating coreset...'):
      if self.min_distances is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)

    self.already_selected = already_selected

    return new_batch

def coreset(vector_ids, already_selected, number_of_datapoints, transformer, distance='euclidean'):

  with tempfile.TemporaryDirectory() as tmpdirname:
    batch_shapes = {}
    for batch_start in range(0, len(vector_ids), BATCH_SIZE):
      features_batch = transformer(vector_ids[batch_start:batch_start + BATCH_SIZE])
      try:
        first_none_vector_idx = features_batch.index(None)
        raise ValueError(f'Cannot use None embeddings for id {vector_ids[first_none_vector_idx + batch_start]} to calculate coreset')
      except ValueError:
        pass

      features_batch = [np.array(f, dtype='float16') for f in features_batch]
      for index, uuid in enumerate(vector_ids[batch_start:batch_start + BATCH_SIZE]):
        vector_shape = features_batch[index].shape
        fp = np.memmap(os.path.join(tmpdirname, str(uuid)), dtype='float16', mode='w+', shape=features_batch[index].shape)
        fp[:] = features_batch[index][:]
        fp.flush()
      try:
        np_features_batch = np.stack(features_batch, axis=0)
      except ValueError as e:
        raise ValueError(f'Cannot use provided embeddings to calculate coreset: {e}') from e

      bfp = np.memmap(
        os.path.join(tmpdirname, str(batch_start)), dtype='float16', mode='w+', shape=np_features_batch.shape)
      bfp[:] = np_features_batch[:]
      bfp.flush()
      batch_shapes[batch_start] = np_features_batch.shape

    X = np.array(vector_ids)

    return kCenterGreedy(
        X=X,
        metric=distance,
        cache_dir=tmpdirname,
        vector_shape=vector_shape,
        batch_shapes=batch_shapes
    ).select_batch_(
        already_selected=already_selected,
        N=min(number_of_datapoints, X.size)
    )