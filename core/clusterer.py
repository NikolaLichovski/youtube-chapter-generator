# clusterer.py
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TranscriptClusterer:
    def __init__(self):
        self.cluster_labels = None
        self.n_clusters = None

    def cluster_segments(self, segments: List[Dict], embedder, n_clusters: Optional[int] = None, method: str = 'kmeans') -> List[Dict]:
        if not segments or len(segments) < 2:
            return segments

        segment_texts = [seg['text'] for seg in segments]
        embeddings = embedder.embed_texts(segment_texts)
        if embeddings is None:
            logger.error("Embedding failed in clustering")
            return segments

        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(embeddings)

        n_clusters = min(n_clusters, len(segments))

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            logger.error(f"Unknown clustering method: {method}")
            return segments

        try:
            cluster_labels = clusterer.fit_predict(embeddings)
            self.cluster_labels = cluster_labels
            self.n_clusters = n_clusters

            clustered_segments = []
            for i, segment in enumerate(segments):
                segment_copy = segment.copy()
                segment_copy['cluster'] = int(cluster_labels[i])
                clustered_segments.append(segment_copy)

            return clustered_segments
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return segments

    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        n_samples = len(embeddings)
        max_clusters = min(max_clusters, n_samples - 1)
        if max_clusters < 2:
            return 2

        best_score = -1
        best_k = 2
        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.warning(f"Silhouette failed for k={k}: {str(e)}")
                continue
        return best_k

    def reorder_clusters_by_time(self, clustered_segments: List[Dict]) -> List[Dict]:
        if not clustered_segments:
            return clustered_segments

        clusters = {}
        for seg in clustered_segments:
            clusters.setdefault(seg['cluster'], []).append(seg)

        cluster_start_times = {cid: min(s['start'] for s in segs) for cid, segs in clusters.items()}
        sorted_cluster_ids = sorted(cluster_start_times, key=cluster_start_times.get)

        reordered_segments = []
        for new_id, old_id in enumerate(sorted_cluster_ids):
            for seg in clusters[old_id]:
                seg_copy = seg.copy()
                seg_copy['cluster'] = new_id
                reordered_segments.append(seg_copy)

        reordered_segments.sort(key=lambda x: x['start'])
        return reordered_segments
