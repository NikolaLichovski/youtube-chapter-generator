import json
import csv
from typing import List, Dict, Optional
from utils.helpers import seconds_to_timestamp
import logging

logger = logging.getLogger(__name__)

class ChapterFormatter:
    def __init__(self):
        pass
    
    def format_chapters(self, titled_segments: List[Dict], 
                       format_type: str = 'youtube') -> str:
        """
        Format chapters for different output types.
        
        Args:
            titled_segments: List of segments with titles
            format_type: 'youtube', 'json', 'markdown', 'csv'
            
        Returns:
            Formatted chapter string
        """
        if not titled_segments:
            return ""
        
        if format_type == 'youtube':
            return self._format_youtube_chapters(titled_segments)
        elif format_type == 'json':
            return self._format_json_chapters(titled_segments)
        elif format_type == 'markdown':
            return self._format_markdown_chapters(titled_segments)
        elif format_type == 'csv':
            return self._format_csv_chapters(titled_segments)
        else:
            return self._format_youtube_chapters(titled_segments)
    
    def _format_youtube_chapters(self, segments: List[Dict]) -> str:
        """Format chapters for YouTube description."""
        chapters = []
        for segment in segments:
            timestamp = seconds_to_timestamp(segment['start'])
            title = segment.get('title', 'Untitled Chapter')
            chapters.append(f"{timestamp} {title}")
        
        return "\n".join(chapters)
    
    def _format_json_chapters(self, segments: List[Dict]) -> str:
        """Format chapters as JSON."""
        chapters = []
        for i, segment in enumerate(segments):
            chapter = {
                'chapter': i + 1,
                'title': segment.get('title', 'Untitled Chapter'),
                'start_time': segment['start'],
                'end_time': segment.get('end', segment['start']),
                'timestamp': seconds_to_timestamp(segment['start']),
                'duration': segment.get('end', segment['start']) - segment['start'],
                'text_preview': segment['text'][:100] + '...' if len(segment['text']) > 100 else segment['text']
            }
            chapters.append(chapter)
        
        return json.dumps({
            'video_chapters': chapters,
            'total_chapters': len(chapters),
            'total_duration': segments[-1].get('end', segments[-1]['start']) if segments else 0
        }, indent=2)
    
    def _format_markdown_chapters(self, segments: List[Dict]) -> str:
        """Format chapters as Markdown."""
        md_content = ["# Video Chapters\n"]
        
        for i, segment in enumerate(segments, 1):
            timestamp = seconds_to_timestamp(segment['start'])
            title = segment.get('title', 'Untitled Chapter')
            duration = segment.get('end', segment['start']) - segment['start']
            
            md_content.append(f"## Chapter {i}: {title}")
            md_content.append(f"**Time:** {timestamp} (Duration: {seconds_to_timestamp(duration)})")
            md_content.append(f"**Preview:** {segment['text'][:150]}...")
            md_content.append("")
        
        return "\n".join(md_content)
    
    def _format_csv_chapters(self, segments: List[Dict]) -> str:
        """Format chapters as CSV."""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Chapter', 'Title', 'Start_Time', 'End_Time', 'Timestamp', 'Duration_Seconds', 'Text_Preview'])
        
        # Data
        for i, segment in enumerate(segments, 1):
            writer.writerow([
                i,
                segment.get('title', 'Untitled Chapter'),
                segment['start'],
                segment.get('end', segment['start']),
                seconds_to_timestamp(segment['start']),
                segment.get('end', segment['start']) - segment['start'],
                segment['text'][:200] + '...' if len(segment['text']) > 200 else segment['text']
            ])
        
        return output.getvalue()
    
    def get_chapter_summary(self, titled_segments: List[Dict]) -> Dict:
        """Get summary statistics about the chapters."""
        if not titled_segments:
            return {}
        
        total_duration = titled_segments[-1].get('end', titled_segments[-1]['start']) if titled_segments else 0
        avg_duration = total_duration / len(titled_segments) if titled_segments else 0
        
        return {
            'total_chapters': len(titled_segments),
            'total_duration_seconds': total_duration,
            'total_duration_formatted': seconds_to_timestamp(total_duration),
            'average_chapter_duration': avg_duration,
            'average_chapter_duration_formatted': seconds_to_timestamp(avg_duration),
            'shortest_chapter': min(seg.get('end', seg['start']) - seg['start'] for seg in titled_segments),
            'longest_chapter': max(seg.get('end', seg['start']) - seg['start'] for seg in titled_segments),
        }
