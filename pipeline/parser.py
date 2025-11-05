"""
Input parser module for the AI Video Generator.
Handles parsing of input JSON files containing video segment definitions.
"""


from typing import List, Optional
from utils.logger import setup_logger
from utils.helpers import load_json

logger = setup_logger(__name__)

class VideoSegment:
    """
    Represents a single video segment with all metadata.
    """
    def __init__(
        self,
        start_time: float,
        end_time: float,
        duration: float,
        transcript: str,
        topic: Optional[str] = None,
        content_type: Optional[str] = None,
        visual_query: Optional[str] = None,
        visual_description: Optional[str] = None,
        reasoning: Optional[str] = None
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.transcript = transcript
        self.topic = topic
        self.content_type = content_type
        self.visual_query = visual_query
        self.visual_description = visual_description
        self.reasoning = reasoning

        # For downstream pipeline
        self.images: List[str] = []          # candidate images
        self.selected_image: str = ""        # best-ranked image
        self.processed_path: str = ""        # processed segment video
        self.subtitles: List[str] = []       # optional subtitles

        # Entity detection results (populated by content-aware effects director)
        self.detected_locations: List[str] = []
        self.detected_persons: List[str] = []
        self.is_first_location_mention: bool = False
        self.is_first_person_mention: bool = False
        self.first_mentioned_location: Optional[str] = None
        self.first_mentioned_person: Optional[str] = None

def parse_input(input_json_path: str) -> List[VideoSegment]:
    """
    Parse input JSON file into a list of video segments.
    
    Args:
        input_json_path (str): Path to the input JSON file
        
    Returns:
        List[VideoSegment]: List of parsed video segments
    """
    logger.info(f"Parsing input file: {input_json_path}")
    data = load_json(input_json_path)
    segments = []

    for seg in data.get("segments", []):
        segment = VideoSegment(
            start_time=seg.get("start_time", 0.0),
            end_time=seg.get("end_time", 0.0),
            duration=seg.get("duration", 5.0),
            transcript=seg.get("transcript", ""),
            topic=seg.get("topic"),
            content_type=seg.get("content_type"),
            visual_query=seg.get("visual_query"),
            visual_description=seg.get("visual_description"),
            reasoning=seg.get("reasoning")
        )
        segments.append(segment)

    logger.info(f"Parsed {len(segments)} segments")
    return segments, data


def extract_script_context(data: dict) -> str:
    """
    Extract overarching theme/intent from the JSON script by analyzing segment topics,
    content types, and transcripts.
    
    Args:
        data (dict): Parsed JSON data
        
    Returns:
        str: Contextual summary of the script's theme and intent
    """
    segments = data.get("segments", [])
    
    # Collect topics and content types
    topics = [seg.get("topic", "") for seg in segments if seg.get("topic")]
    content_types = [seg.get("content_type", "") for seg in segments if seg.get("content_type")]
    
    # Get first few transcripts for narrative context
    opening_transcripts = " ".join([
        seg.get("transcript", "") for seg in segments[:3]
    ])[:400]
    
    # Extract generation metadata
    generation_method = data.get("generation_method", "")
    
    # Build context summary
    context_parts = []
    
    if opening_transcripts:
        context_parts.append(f"Opening narrative: {opening_transcripts}")
    
    if topics:
        unique_topics = list(dict.fromkeys(topics))[:5]  # First 5 unique topics
        context_parts.append(f"Key topics: {', '.join(unique_topics)}")
    
    if content_types:
        unique_types = list(dict.fromkeys(content_types))
        context_parts.append(f"Content style: {', '.join(unique_types)}")
    
    if generation_method:
        context_parts.append(f"Generation approach: {generation_method}")
    
    return "\n".join(context_parts)