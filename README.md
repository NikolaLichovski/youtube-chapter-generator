# 🎬 YouTube Chapter Generator

An AI-powered tool that automatically generates timestamped chapters for YouTube videos using transcript analysis, semantic segmentation, and natural language processing.

## ✨ Features

- **Automatic Transcript Extraction**: Fetches transcripts from YouTube videos with captions
- **Intelligent Segmentation**: Multiple methods including semantic similarity, time-based, and length-based segmentation
- **AI-Powered Title Generation**: Uses extractive and abstractive methods to generate meaningful chapter titles
- **Multiple Export Formats**: YouTube description format, JSON, Markdown, and CSV
- **Interactive Timeline**: Visual representation of generated chapters
- **Customizable Parameters**: Adjust number of chapters, segmentation methods, and title generation approaches
- **Real-time Editing**: Edit generated chapter titles before export

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NikolaLichovski/youtube-chapter-generator.git
cd youtube-chapter-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## 📋 Usage

1. **Enter YouTube URL**: Paste the URL of a YouTube video with captions enabled
2. **Configure Settings**: Choose your preferred segmentation method and title generation approach
3. **Generate Chapters**: Click the "Generate Chapters" button to process the video
4. **Review & Edit**: Review generated chapters and edit titles if needed
5. **Export**: Download chapters in your preferred format

### Supported Video Requirements

- Video must have captions/subtitles enabled (auto-generated or manual)
- Video must be publicly accessible
- No age restrictions or region blocks

## 🧠 How It Works

### 1. Transcript Extraction
Uses the `youtube-transcript-api` to fetch transcript data with timing information.

### 2. Text Segmentation
Three segmentation methods available:
- **Semantic**: Uses sentence embeddings and clustering to group related content
- **Time-based**: Splits content into fixed time intervals
- **Length-based**: Segments by word count

### 3. Title Generation
Multiple approaches:
- **Extractive**: Uses RAKE algorithm to extract key phrases
- **Abstractive**: Uses transformer models (BART) for summary-based titles
- **Hybrid**: Combines both methods for best results

### 4. Output Formatting
Generates chapters in multiple formats suitable for different use cases.

## ⚙️ Configuration Options

### Embedding Models
- `all-MiniLM-L6-v2`: Fast and efficient (default)
- `all-mpnet-base-v2`: Higher quality, slower
- `distilbert-base-nli-stsb-mean-tokens`: Alternative option

### Segmentation Parameters
- **Number of chapters**: 2-20 chapters
- **Segment duration**: 30-600 seconds (time-based)
- **Words per segment**: 50-500 words (length-based)

### Title Generation Methods
- **Extractive**: Fast, keyword-based titles
- **Abstractive**: AI-generated summaries as titles
- **Hybrid**: Best of both methods

## 📊 Export Formats

1. **YouTube Description Format**: Ready to paste into video descriptions
   ```
   00:00 Introduction
   02:30 Main Topic Discussion
   05:45 Key Points Summary
   ```

2. **JSON**: Structured data with metadata
3. **Markdown**: Formatted documentation with previews
4. **CSV**: Spreadsheet-compatible format

## 🛠️ Project Structure

```
youtube-chapter-generator/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── core/                    # Core functionality modules
│   ├── transcript_loader.py # YouTube transcript extraction
│   ├── segmenter.py         # Text segmentation logic
│   ├── embedder.py          # Text embedding generation
│   ├── clusterer.py         # Semantic clustering
│   ├── summarizer.py        # Title generation
│   └── formatter.py         # Output formatting
└── utils/                   # Utility functions
    └── helpers.py           # Helper functions
```

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **youtube-transcript-api**: YouTube transcript extraction
- **sentence-transformers**: Text embeddings
- **transformers**: Hugging Face transformer models
- **scikit-learn**: Machine learning utilities
- **nltk**: Natural language processing
- **plotly**: Interactive visualizations

### Machine Learning Models
- **Sentence Transformers**: For semantic text embeddings
- **BART**: For abstractive summarization
- **KMeans/Hierarchical Clustering**: For semantic segmentation
- **RAKE**: For keyword extraction


## ⚠️ Limitations

- Requires videos to have captions enabled
- Processing time depends on video length and chosen models
- Quality depends on transcript accuracy
- Some models require internet connection for initial download

## 🆘 Troubleshooting

### Common Issues

1. **"No transcript found"**: Ensure the video has captions enabled
2. **Slow processing**: Try using lighter models or shorter videos
3. **Poor chapter titles**: Try different title generation methods
4. **Memory issues**: Reduce number of chapters or use time-based segmentation

