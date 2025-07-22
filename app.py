import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import logging
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from core.transcript_loader import TranscriptLoader
from core.embedder import TextEmbedder
from core.segmenter import TranscriptSegmenter
from core.clusterer import TranscriptClusterer
from core.summarizer import ChapterTitleGenerator
from core.formatter import ChapterFormatter
from utils.helpers import extract_video_id, seconds_to_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="YouTube Chapter Generator",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)


class YouTubeChapterGenerator:
    def __init__(self):
        self.transcript_loader = TranscriptLoader()
        self.embedder = None
        self.segmenter = None
        self.clusterer = TranscriptClusterer()
        self.title_generator = None
        self.formatter = ChapterFormatter()

        # Initialize session state
        if 'video_data' not in st.session_state:
            st.session_state.video_data = {}
        if 'chapters' not in st.session_state:
            st.session_state.chapters = []
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False

    def initialize_models(self, embedding_model: str, title_method: str):
        """Initialize the required models."""
        try:
            with st.spinner("Loading models..."):
                self.embedder = TextEmbedder(embedding_model)
                self.segmenter = TranscriptSegmenter(self.embedder)
                self.title_generator = ChapterTitleGenerator(title_method)
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            logger.error(f"Error loading models: {str(e)}")
            return False

    def _normalize_transcript_data(self, transcript_data):
        """
        Convert transcript data to consistent dictionary format.

        Args:
            transcript_data: Raw transcript data (could be FetchedTranscriptSnippet objects or dicts)

        Returns:
            List of dictionaries with 'text', 'start', and 'duration' keys
        """
        normalized_data = []

        for item in transcript_data:
            try:
                # If it's already a dictionary
                if isinstance(item, dict):
                    normalized_item = {
                        'text': item.get('text', ''),
                        'start': item.get('start', 0),
                        'duration': item.get('duration', 0)
                    }
                else:
                    # If it's a FetchedTranscriptSnippet or similar object
                    normalized_item = {
                        'text': getattr(item, 'text', ''),
                        'start': getattr(item, 'start', 0),
                        'duration': getattr(item, 'duration', 0)
                    }

                normalized_data.append(normalized_item)

            except Exception as e:
                logger.warning(f"Error normalizing transcript item: {str(e)}")
                # Add a default entry to avoid breaking the process
                normalized_data.append({
                    'text': '',
                    'start': 0,
                    'duration': 0
                })

        return normalized_data

    def process_video(self, video_url: str, segmentation_method: str,
                      num_chapters: int, **kwargs) -> bool:
        try:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL.")
                return False

            with st.spinner("Loading transcript..."):
                raw_transcript_data = self.transcript_loader.load_transcript(video_id)
                if not raw_transcript_data:
                    st.error("Transcript not found or captions are disabled.")
                    return False

            transcript_data = self._normalize_transcript_data(raw_transcript_data)
            st.success(f"Loaded transcript with {len(transcript_data)} segments")

            # Segment transcript
            with st.spinner("Segmenting transcript..."):
                if segmentation_method == "Time-based":
                    # For 30min video, use 3-4 min segments
                    segment_duration = min(kwargs.get('segment_duration', 180), 300)
                    segments = self.segmenter.segment_by_time(transcript_data, segment_duration)
                elif segmentation_method == "Semantic":
                    # Limit clusters for better quality
                    effective_chapters = min(num_chapters, 8)  # Max 8 chapters
                    segments = self.segmenter.segment_by_semantic_similarity(
                        transcript_data, effective_chapters
                    )
                elif segmentation_method == "Length-based":
                    # Use smaller segments for better titles
                    words_per_segment = min(kwargs.get('words_per_segment', 200), 300)
                    segments = self.segmenter.segment_by_length(transcript_data, words_per_segment)
                else:
                    segments = self.segmenter.segment_by_time(transcript_data)

            if not segments:
                st.error("Segmentation failed.")
                return False

            st.success(f"Created {len(segments)} segments")

            # Run clustering on segments
            with st.spinner("Clustering segments..."):
                clustered_segments = self.clusterer.cluster_segments(segments, self.embedder, num_chapters)
                clustered_segments = self.clusterer.reorder_clusters_by_time(clustered_segments)
                # # DEBUG: Print cluster information
                # st.write(f"DEBUG: Created {len(set(seg['cluster'] for seg in clustered_segments))} clusters")
                # for cluster_id in sorted(set(seg['cluster'] for seg in clustered_segments)):
                #     cluster_segs = [seg for seg in clustered_segments if seg['cluster'] == cluster_id]
                #     combined_text = ' '.join([seg['text'] for seg in cluster_segs])
                #     st.write(f"Cluster {cluster_id}: {len(cluster_segs)} segments, {len(combined_text)} chars")
                #     st.write(f"Sample text: {combined_text[:200]}...")

            # Generate one title per cluster
            with st.spinner("Generating chapter titles..."):
                cluster_titles = self.title_generator.generate_titles_for_clusters(clustered_segments)
                # # DEBUG: Print generated titles
                # st.write("DEBUG: Generated titles:")
                # for cluster_id, title in cluster_titles.items():
                #     st.write(f"Cluster {cluster_id}: '{title}'")

                # Choose the earliest segment in each cluster to represent the chapter
                cluster_map = {}
                for seg in clustered_segments:
                    cluster_id = seg['cluster']
                    if cluster_id not in cluster_map or seg['start'] < cluster_map[cluster_id]['start']:
                        cluster_map[cluster_id] = seg

                titled_segments = []
                for cluster_id, rep_seg in sorted(cluster_map.items()):
                    seg_copy = rep_seg.copy()
                    seg_copy['title'] = cluster_titles.get(cluster_id, f"Chapter {cluster_id + 1}")
                    titled_segments.append(seg_copy)

            st.success("Generated chapter titles")

            # Save state
            st.session_state.video_data = {
                'video_id': video_id,
                'video_url': video_url,
                'transcript_data': transcript_data,
                'segments': clustered_segments
            }
            st.session_state.chapters = titled_segments
            st.session_state.processing_complete = True

            return True

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Error processing video: {str(e)}")
            return False

    def display_chapters(self):
        """Display the generated chapters."""
        if not st.session_state.chapters:
            return

        chapters = st.session_state.chapters

        # Display summary
        try:
            summary = self.formatter.get_chapter_summary(chapters)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chapters", summary.get('total_chapters', 0))
            with col2:
                st.metric("Total Duration", summary.get('total_duration_formatted', '00:00'))
            with col3:
                st.metric("Avg Chapter Length", summary.get('average_chapter_duration_formatted', '00:00'))
            with col4:
                st.metric("Video ID", st.session_state.video_data.get('video_id', 'N/A'))
        except Exception as e:
            st.warning(f"Could not generate summary: {str(e)}")

        # Timeline visualization
        st.subheader("📊 Chapter Timeline")
        self.create_timeline_chart(chapters)

        # Chapter list
        st.subheader("📋 Chapter List")

        # Allow editing of chapter titles
        edited_chapters = []
        for i, chapter in enumerate(chapters):
            try:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_title = st.text_input(
                            f"Chapter {i + 1} Title",
                            value=chapter.get('title', f'Chapter {i + 1}'),
                            key=f"title_{i}"
                        )
                    with col2:
                        timestamp = seconds_to_timestamp(chapter.get('start', 0))
                        st.text(f"⏰ {timestamp}")

                    # Show text preview
                    with st.expander(f"Preview - {new_title}"):
                        chapter_text = chapter.get('text', 'No text available')
                        preview_text = chapter_text[:300] + '...' if len(chapter_text) > 300 else chapter_text
                        st.text(preview_text)

                    # Update chapter with new title
                    updated_chapter = chapter.copy()
                    updated_chapter['title'] = new_title
                    edited_chapters.append(updated_chapter)

            except Exception as e:
                st.error(f"Error displaying chapter {i + 1}: {str(e)}")
                # Add a default chapter to prevent breaking
                edited_chapters.append({
                    'title': f'Chapter {i + 1}',
                    'start': chapter.get('start', 0),
                    'text': chapter.get('text', 'No text available')
                })

        st.session_state.chapters = edited_chapters

    def create_timeline_chart(self, chapters: List[Dict]):
        """Create a timeline chart of chapters."""
        try:
            df_data = []
            for i, chapter in enumerate(chapters):
                start_time = chapter.get('start', 0)
                end_time = chapter.get('end', start_time + 60)  # Default 1 min if no end

                df_data.append({
                    'Chapter': f"Ch {i + 1}",
                    'Title': chapter.get('title', f'Chapter {i + 1}'),
                    'Start': start_time,
                    'End': end_time,
                    'Duration': end_time - start_time
                })

            if not df_data:
                st.warning("No data available for timeline chart")
                return

            df = pd.DataFrame(df_data)

            # Create Gantt-style chart
            fig = px.timeline(
                df,
                x_start='Start',
                x_end='End',
                y='Chapter',
                text='Title',
                title="Chapter Timeline",
                color='Duration',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                height=max(400, len(chapters) * 50),
                xaxis_title="Time (seconds)",
                yaxis_title="Chapters"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating timeline chart: {str(e)}")
            logger.error(f"Error creating timeline chart: {str(e)}")

    def export_chapters(self):
        """Export chapters in different formats."""
        if not st.session_state.chapters:
            st.warning("No chapters to export. Please process a video first.")
            return

        st.subheader("📥 Export Chapters")

        format_type = st.selectbox(
            "Choose export format:",
            ["YouTube Description", "JSON", "Markdown", "CSV"]
        )

        format_mapping = {
            "YouTube Description": "youtube",
            "JSON": "json",
            "Markdown": "markdown",
            "CSV": "csv"
        }

        try:
            formatted_content = self.formatter.format_chapters(
                st.session_state.chapters,
                format_mapping[format_type]
            )

            st.text_area(
                f"Formatted content ({format_type}):",
                formatted_content,
                height=300
            )

            # Download button
            file_extension = format_mapping[format_type] if format_mapping[format_type] != 'youtube' else 'txt'
            st.download_button(
                label=f"Download {format_type}",
                data=formatted_content,
                file_name=f"chapters_{st.session_state.video_data.get('video_id', 'video')}.{file_extension}",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error formatting chapters: {str(e)}")
            logger.error(f"Error formatting chapters: {str(e)}")


def main():
    """Main Streamlit application."""

    # Header
    st.title("🎬 YouTube Chapter Generator")
    st.markdown("Generate timestamped chapters for YouTube videos automatically using AI")

    # Initialize the generator
    generator = YouTubeChapterGenerator()

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model settings
    embedding_model = st.sidebar.selectbox(
        "Embedding Model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "distilbert-base-nli-stsb-mean-tokens"],
        index=0
    )

    title_method = st.sidebar.selectbox(
        "Title Generation Method:",
        ["extractive", "abstractive", "hybrid"],
        index=0
    )

    # Segmentation settings
    st.sidebar.subheader("📝 Segmentation Settings")
    segmentation_method = st.sidebar.selectbox(
        "Segmentation Method:",
        ["Semantic", "Time-based", "Length-based"],
        index=0
    )

    num_chapters = st.sidebar.number_input(
        "Target Number of Chapters:",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )

    # Additional parameters based on segmentation method
    segment_duration = 120
    words_per_segment = 150

    if segmentation_method == "Time-based":
        segment_duration = st.sidebar.number_input(
            "Segment Duration (seconds):",
            min_value=30,
            max_value=600,
            value=120,
            step=30
        )
    elif segmentation_method == "Length-based":
        words_per_segment = st.sidebar.number_input(
            "Words per Segment:",
            min_value=50,
            max_value=500,
            value=150,
            step=25
        )

    # Main interface
    video_url = st.text_input(
        "🔗 Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the full YouTube URL here"
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        process_button = st.button("🚀 Generate Chapters", type="primary")

    with col2:
        if st.session_state.processing_complete:
            st.success("✅ Processing complete!")

    # Process video when button is clicked
    if process_button and video_url:
        # Reset processing state
        st.session_state.processing_complete = False
        st.session_state.chapters = []

        # Initialize models
        if generator.initialize_models(embedding_model, title_method):
            # Process the video
            success = generator.process_video(
                video_url=video_url,
                segmentation_method=segmentation_method,
                num_chapters=num_chapters,
                segment_duration=segment_duration,
                words_per_segment=words_per_segment
            )

            if success:
                st.balloons()

    # Display results if processing is complete
    if st.session_state.processing_complete:
        st.markdown("---")

        # Create tabs for different views
        tab1, tab2 = st.tabs(["📋 Chapters", "📥 Export"])

        with tab1:
            generator.display_chapters()

        with tab2:
            generator.export_chapters()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Built with ❤️ using Streamlit, HuggingFace Transformers, and Sentence Transformers</p>
            <p>⚠️ Note: This tool requires videos to have captions/subtitles enabled</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()