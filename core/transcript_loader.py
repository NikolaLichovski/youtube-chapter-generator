from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptLoader:
    def __init__(self):
        self.transcript_data = []
        self.api = YouTubeTranscriptApi()  # create an instance once here

    def load_transcript(self, video_id: str, languages: List[str] = ['en']) -> Optional[List[Dict]]:
        """
        Load transcript from YouTube video.

        Args:
            video_id: YouTube video ID
            languages: List of preferred languages

        Returns:
            List of transcript segments with text, start, and duration
        """
        try:
            # Use the old instance method .list()
            transcript_list = self.api.list(video_id)

            # Try manual transcripts first
            for lang in languages:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    self.transcript_data = transcript.fetch()
                    logger.info(f"Found manual transcript in {lang}")
                    return self.transcript_data
                except NoTranscriptFound:
                    continue

            # Try auto-generated transcripts
            for lang in languages:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    self.transcript_data = transcript.fetch()
                    logger.info(f"Found auto-generated transcript in {lang}")
                    return self.transcript_data
                except NoTranscriptFound:
                    continue

            # If no transcript in preferred languages, try to get any available transcript
            try:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    first_transcript = available_transcripts[0]
                    self.transcript_data = first_transcript.fetch()
                    logger.info(f"Found transcript in language: {first_transcript.language}")
                    return self.transcript_data
            except Exception as e:
                logger.error(f"Error accessing available transcripts: {str(e)}")

        except TranscriptsDisabled:
            logger.error("Transcripts are disabled for this video")
            return None
        except NoTranscriptFound:
            logger.error("No transcript found for this video")
            return None
        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            return None

        logger.error("No transcripts available for this video")
        return None

    def get_full_text(self) -> str:
        """
        Get full transcript text as a single string.
        """
        if not self.transcript_data:
            return ""

        try:
            return " ".join([item['text'] for item in self.transcript_data])
        except (KeyError, TypeError) as e:
            logger.error(f"Error extracting text from transcript data: {str(e)}")
            return ""

    def get_timed_segments(self) -> List[Dict]:
        """
        Get transcript segments with timing information.
        """
        return self.transcript_data if self.transcript_data else []
