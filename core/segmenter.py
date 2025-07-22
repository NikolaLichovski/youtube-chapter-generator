import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import logging

nltk.download('punkt', quiet=True)
logger = logging.getLogger(__name__)


class TranscriptSegmenter:
    def __init__(self, embedder):
        self.embedder = embedder

    def segment_by_semantic_similarity(self, transcript_data: List[Dict], num_segments: int = 5,
                                       window_size: int = 10) -> List[Dict]:
        """Fixed semantic segmentation with better boundary detection."""
        if not transcript_data or len(transcript_data) < num_segments:
            return self.segment_by_time(transcript_data, segment_duration=180)  # 3min fallback

        # Create overlapping windows of transcript items (not sentences)
        total_items = len(transcript_data)
        window_size = max(10, total_items // (num_segments * 3))  # Adaptive window size

        windows = []
        window_starts = []

        # Create overlapping windows
        for i in range(0, total_items, window_size // 2):  # 50% overlap
            end_idx = min(i + window_size, total_items)
            if end_idx - i < window_size // 2:  # Skip too small windows
                break

            window_text = ' '.join([item['text'] for item in transcript_data[i:end_idx]])
            windows.append(window_text)
            window_starts.append(i)

            if end_idx >= total_items:
                break

        if len(windows) < num_segments:
            return self.segment_by_time(transcript_data, segment_duration=180)

        # Get embeddings for windows
        embeddings = self.embedder.embed_texts(windows)
        if embeddings is None:
            logger.warning("Embedding failed in semantic segmentation, falling back to time segmentation.")
            return self.segment_by_time(transcript_data, segment_duration=180)

        # Calculate similarities between adjacent windows
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        # Find the lowest similarity points as boundaries (topic changes)
        boundary_indices = sorted(np.argsort(similarities)[:num_segments - 1])

        # Convert window boundaries back to transcript item boundaries
        segments = []
        start_idx = 0

        for boundary_idx in boundary_indices:
            # Map window boundary to transcript item boundary
            end_idx = window_starts[boundary_idx + 1] if boundary_idx + 1 < len(window_starts) else total_items

            if end_idx <= start_idx:
                continue

            segment_items = transcript_data[start_idx:end_idx]

            if segment_items:
                segments.append({
                    'start': segment_items[0]['start'],
                    'end': segment_items[-1]['start'] + segment_items[-1].get('duration', 0),
                    'text': ' '.join([item['text'] for item in segment_items]),
                    'items': segment_items
                })

            start_idx = end_idx

        # Add final segment
        if start_idx < total_items:
            segment_items = transcript_data[start_idx:]
            segments.append({
                'start': segment_items[0]['start'],
                'end': segment_items[-1]['start'] + segment_items[-1].get('duration', 0),
                'text': ' '.join([item['text'] for item in segment_items]),
                'items': segment_items
            })

        # Ensure we have reasonable segments
        if not segments or len(segments) < 2:
            return self.segment_by_time(transcript_data, segment_duration=300)  # 5min fallback

        return segments

    def segment_by_time(self, transcript_data: List[Dict], segment_duration: float = 120.0) -> List[Dict]:
        """Improved time-based segmentation."""
        if not transcript_data:
            return []

        segments = []
        current_segment = {'start': transcript_data[0]['start'], 'text': '', 'items': []}

        for item in transcript_data:
            # If current segment would be too long, start a new one
            if item['start'] - current_segment['start'] >= segment_duration:
                if current_segment['text'].strip():
                    current_segment['end'] = current_segment['items'][-1]['start'] + current_segment['items'][-1].get(
                        'duration', 0)
                    segments.append(current_segment)
                current_segment = {'start': item['start'], 'text': '', 'items': []}

            current_segment['text'] += ' ' + item['text']
            current_segment['items'].append(item)

        # Add final segment
        if current_segment['text'].strip():
            current_segment['end'] = current_segment['items'][-1]['start'] + current_segment['items'][-1].get(
                'duration', 0)
            segments.append(current_segment)

        return segments

    def segment_by_length(self, transcript_data: List[Dict], words_per_segment: int = 150) -> List[Dict]:
        """Fixed length-based segmentation."""
        if not transcript_data:
            return []

        segments = []
        current_segment = {'start': transcript_data[0]['start'], 'text': '', 'items': [], 'word_count': 0}

        for item in transcript_data:
            word_count = len(item['text'].split())

            # Start new segment if current would be too long
            if current_segment['word_count'] + word_count > words_per_segment and current_segment['items']:
                current_segment['end'] = current_segment['items'][-1]['start'] + current_segment['items'][-1].get(
                    'duration', 0)
                segments.append(current_segment)
                current_segment = {'start': item['start'], 'text': '', 'items': [], 'word_count': 0}

            current_segment['text'] += ' ' + item['text']
            current_segment['items'].append(item)
            current_segment['word_count'] += word_count

        # Add final segment
        if current_segment['items']:
            current_segment['end'] = current_segment['items'][-1]['start'] + current_segment['items'][-1].get(
                'duration', 0)
            segments.append(current_segment)

        return segments