import re
from typing import List, Dict, Optional
from rake_nltk import Rake
import nltk
import logging
from transformers import pipeline
import torch

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
logger = logging.getLogger(__name__)


class ChapterTitleGenerator:
    def __init__(self, method: str = 'hybrid'):
        self.method = method
        self.summarizer = None
        self.rake = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models with better error handling."""
        try:
            if self.method in ['abstractive', 'hybrid']:
                try:
                    # Use a smaller, more reliable model
                    model_name = 'sshleifer/distilbart-cnn-12-6'
                    self.summarizer = pipeline(
                        'summarization',
                        model=model_name,
                        tokenizer=model_name,
                        device=-1,  # CPU
                        framework='pt'
                    )
                    logger.info(f"Loaded summarization model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load summarizer: {e}")
                    self.method = 'extractive'

            if self.method in ['extractive', 'hybrid'] or self.summarizer is None:
                self.rake = Rake()
                logger.info("Initialized RAKE for extractive method.")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.method = 'extractive'
            self.rake = Rake()

    def generate_titles_for_clusters(self, clustered_segments: List[Dict]) -> Dict[int, str]:
        """Generate titles for clusters with better text processing."""
        if not clustered_segments:
            return {}

        # Group segments by cluster
        cluster_texts = {}
        for segment in clustered_segments:
            cluster_id = segment['cluster']
            if cluster_id not in cluster_texts:
                cluster_texts[cluster_id] = []
            cluster_texts[cluster_id].append(segment['text'])

        # Generate titles
        titles = {}
        for cluster_id, texts in cluster_texts.items():
            combined_text = ' '.join(texts)

            # IMPORTANT: Truncate very long texts
            words = combined_text.split()
            if len(words) > 500:  # Limit to ~500 words for processing
                combined_text = ' '.join(words[:500])

            cleaned_text = self._clean_text(combined_text)
            title = self._generate_title(cleaned_text, cluster_id)
            titles[cluster_id] = title

        return titles

    def _generate_title(self, text: str, index: int) -> str:
        """Generate title with improved logic."""
        if not text.strip():
            return f"Chapter {index + 1}"

        if self.method == 'extractive':
            return self._extractive_title(text, index)
        elif self.method == 'abstractive':
            return self._abstractive_title(text, index)
        elif self.method == 'hybrid':
            # Try abstractive first, fall back to extractive
            title = self._abstractive_title(text, index)
            if not title or title == f"Chapter {index + 1}" or len(title.split()) < 2:
                title = self._extractive_title(text, index)
            return title
        else:
            return self._extractive_title(text, index)

    def _clean_text(self, text: str) -> str:
        """Improved text cleaning for transcript data."""
        if not text or not text.strip():
            return ""

        # Remove transcription artifacts
        text = re.sub(r'\[.*?\]', '', text)  # [MUSIC], [APPLAUSE]
        text = re.sub(r'\(.*?\)', '', text)  # (inaudible)
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove excessive filler words but keep some context
        words = text.split()

        # Common filler words to reduce (not eliminate completely)
        filler_words = {'um', 'uh', 'like', 'you know', 'so', 'well', 'okay', 'alright'}

        # Keep every 3rd filler word to maintain some natural flow
        cleaned_words = []
        filler_count = 0

        for word in words:
            word_lower = word.lower().strip('.,!?')
            if word_lower in filler_words:
                filler_count += 1
                if filler_count % 3 == 0:  # Keep every 3rd filler word
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)

        return ' '.join(cleaned_words)

    def _extractive_title(self, text: str, index: int) -> str:
        """Improved extractive title generation."""
        try:
            if not text.strip():
                return f"Chapter {index + 1}"

            # Method 1: Use RAKE if available
            if self.rake:
                self.rake.extract_keywords_from_text(text)
                phrases = self.rake.get_ranked_phrases()

                if phrases:
                    # Get top phrases and combine intelligently
                    top_phrases = phrases[:3]

                    # Find the best phrase (not too short, not too long)
                    best_phrase = None
                    for phrase in top_phrases:
                        words = phrase.split()
                        if 3 <= len(words) <= 7:  # Good length
                            best_phrase = phrase
                            break

                    if not best_phrase and top_phrases:
                        best_phrase = top_phrases[0]

                    if best_phrase:
                        title = self._clean_title(best_phrase)
                        if title and len(title.split()) >= 2:
                            return title.title()

            # Method 2: Find key topics using simple frequency
            return self._simple_topic_extraction(text, index)

        except Exception as e:
            logger.error(f"Extractive title generation failed: {e}")
            return f"Chapter {index + 1}"

    def _simple_topic_extraction(self, text: str, index: int) -> str:
        """Simple topic extraction as fallback."""
        try:
            # Split into sentences and find the most informative one
            sentences = re.split(r'[.!?]+', text)

            # Score sentences based on meaningful words
            best_sentence = ""
            best_score = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue

                words = sentence.lower().split()

                # Score based on presence of meaningful words
                meaningful_words = [
                    'build', 'create', 'learn', 'understand', 'implement', 'setup', 'configure',
                    'install', 'run', 'execute', 'test', 'debug', 'fix', 'solve', 'explain',
                    'tutorial', 'guide', 'introduction', 'basics', 'advanced', 'tips', 'tricks'
                ]

                score = sum(1 for word in words if any(mw in word for mw in meaningful_words))
                score += len([w for w in words if len(w) > 4])  # Longer words bonus
                score -= len([w for w in words if w in ['um', 'uh', 'like', 'you', 'know']])  # Filler penalty

                if score > best_score:
                    best_score = score
                    best_sentence = sentence

            if best_sentence:
                # Extract key part of the sentence
                words = best_sentence.split()[:8]  # First 8 words
                title = ' '.join(words)
                title = self._clean_title(title)
                if title:
                    return title.title()

            # Final fallback: use most frequent meaningful words
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get top 3-5 most frequent words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_words:
                title_words = [word for word, freq in top_words if freq > 1][:4]
                if title_words:
                    return ' '.join(title_words).title()

            return f"Chapter {index + 1}"

        except Exception as e:
            logger.error(f"Simple topic extraction failed: {e}")
            return f"Chapter {index + 1}"

    def _abstractive_title(self, text: str, index: int) -> str:
        """Fixed abstractive title generation."""
        try:
            if not self.summarizer or not text.strip():
                return self._extractive_title(text, index)

            # Prepare text - limit length for BART
            words = text.split()
            if len(words) > 400:  # Reduced from 512
                text = ' '.join(words[:400])

            # Generate summary with proper parameters
            try:
                summary_result = self.summarizer(
                    text,
                    max_length=25,
                    min_length=5,
                    do_sample=False,
                    truncation=True,
                    clean_up_tokenization_spaces=True
                )

                if summary_result and len(summary_result) > 0:
                    raw_title = summary_result[0]['summary_text']
                    title = self._post_process_abstractive_title(raw_title)

                    if title and len(title.split()) >= 2:
                        return title

            except Exception as model_error:
                logger.warning(f"BART model error: {model_error}")

            # Fallback to extractive
            return self._extractive_title(text, index)

        except Exception as e:
            logger.error(f"Abstractive title generation failed: {e}")
            return self._extractive_title(text, index)

    def _post_process_abstractive_title(self, raw_title: str) -> str:
        """Clean up BART-generated titles."""
        if not raw_title:
            return ""

        title = raw_title.strip()

        # Remove common BART artifacts
        prefixes_to_remove = [
            r'^(the\s+)?(video|tutorial|guide|chapter)\s+(shows?|explains?|demonstrates?|teaches?)\s+',
            r'^(this\s+)?(video|tutorial|guide|chapter)\s+',
            r'^(in\s+this\s+)?(video|tutorial|guide|chapter)\s*,?\s*',
            r'^(here\s+)?(we|you|i)\s+(will\s+)?(learn|see|discover|explore)\s+',
            r'\.+$',
        ]

        for pattern in prefixes_to_remove:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()

        # Clean and format
        title = self._clean_title(title)

        if title:
            # Proper title case
            words = title.split()
            if words:
                # Capitalize first word and important words
                result = [words[0].capitalize()]

                for word in words[1:]:
                    if word.lower() in ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                                        'with']:
                        result.append(word.lower())
                    else:
                        result.append(word.capitalize())

                title = ' '.join(result)

        return title

    def _clean_title(self, title: str) -> str:
        """Clean title text."""
        if not title:
            return ""

        # Remove punctuation except hyphens
        title = re.sub(r'[^\w\s\-]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()

        # Remove filler words from start
        filler_starts = ['so', 'well', 'now', 'okay', 'alright', 'um', 'uh', 'like']
        words = title.lower().split()

        while words and words[0] in filler_starts:
            words = words[1:]

        title = ' '.join(words)
        return title if len(title) >= 5 else ""  # Minimum length check
