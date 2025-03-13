#!pip install openai-whisper
import ipywidgets as widgets
from IPython.display import display
import requests
import os
from urllib.parse import urlparse
import whisper
from dataclasses import dataclass
import json


# Configuration - Change these values as needed
file = ""  # Your audio file
model_size = "large-v3-turbo"  # Options: "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
language = None     # Set to language code (e.g., "en") or None for auto-detection
device = "cuda"     # Use "cuda" for GPU, "cpu" for CPU
max_words_per_segment = 10  # Maximum words per segment

# Create a text input for the URL
url_input = widgets.Text(
    value='',
    placeholder='Enter file URL',
    description='URL:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='70%')
)

# Create a button with label "vykdyti"
download_button = widgets.Button(
    description='vykdyti',
    button_style='success', 
    tooltip='Download file from URL',
    layout=widgets.Layout(width='100px')
)

# Create an output widget to display status messages
output = widgets.Output()

# Function to download the file
def download_file(b):
    global file
    with output:
        output.clear_output()
        url = url_input.value.strip()
        
        if not url:
            print("Please enter a URL")
            return
            
        try:
            print(f"Downloading from: {url}")
            
            # Send request to get the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Extract filename from URL or use Content-Disposition header if available
            if "Content-Disposition" in response.headers:
                content_disposition = response.headers["Content-Disposition"]
                filename = content_disposition.split("filename=")[1].strip('"')
            else:
                # Extract filename from URL
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                
            # If filename is empty, use a default name
            if not filename:
                filename = "downloaded_file"
                
            # Save the file to current directory
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            print(f"File '{filename}' successfully downloaded to {os.getcwd()}")
            file = filename
            transcribe()
            
        except Exception as e:
            print(f"Error: {str(e)}")





# Simple dataclass to mimic faster-whisper's segment format for our new segments
@dataclass
class SimpleSegment:
    start: float
    end: float
    text: str

# Function to format timestamp for VTT format
def format_timestamp(seconds):
    # VTT format: HH:MM:SS.mmm
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"

# Function to create VTT format from segments
def generate_vtt(segments):
    vtt = "WEBVTT\n\n"
    
    for i, segment in enumerate(segments):
        # Use dictionary access if segment is a dict
        if isinstance(segment, dict):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"]
        else:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text
            
        vtt += f"{start} --> {end}\n{text}\n\n"
    
    return vtt

# Function to split segments using word timestamps
def create_short_segments_with_word_timestamps(segments, max_words):
    new_segments = []
    
    for segment in segments:
        # Skip if segment has no words with timestamps
        if "words" not in segment or not segment["words"]:
            new_segments.append(segment)
            continue
            
        words = segment["words"]
        # Process each chunk of max_words
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i+max_words]
            if not chunk_words:
                continue
                
            # Get start time from first word and end time from last word
            chunk_start = chunk_words[0]["start"]
            chunk_end = chunk_words[-1]["end"]
            
            # Join the words into text
            chunk_text = " ".join(word["word"] for word in chunk_words)
            
            # Create new segment using SimpleSegment
            new_segments.append(SimpleSegment(
                start=chunk_start,
                end=chunk_end,
                text=chunk_text
            ))
    
    return new_segments
    
def transcribe():
    # Load the model
    # Transcribe with word-level timestamps enabled
    
    print(f"Loading {model_size} model on {device}...")
    model = whisper.load_model(model_size, device=device)
    fullpath = os.getcwd() + "/" + file
    print(fullpath)
    # Transcribe
    print("Transcribing...")
    result = model.transcribe(
        fullpath, 
        language=language,
        verbose=False,  # Set to True to see progress
        fp16=True,  # Use fp16 precision on GPU for better performance
        word_timestamps=True 
    )
    #print(result)
    #print(json.dumps(result, indent=2))
    segments = result["segments"]
    
    # Convert segments to list
    segments_list = list(segments)
    #print(f"Original segments: {len(segments_list)}")
    
    # Create shorter segments using word timestamps
    short_segments = create_short_segments_with_word_timestamps(segments_list, max_words_per_segment)
    #print(f"After splitting: {len(short_segments)} segments")
    
    # Generate VTT
    vtt_output = generate_vtt(short_segments)
    
    # Print VTT output
    #print("\n--- VTT FORMAT ---")
    #print(vtt_output)
    
    
    # Optional: Save to file
    with open("transcript.vtt", "w", encoding="utf-8") as f:
        f.write(vtt_output)
    
    
    # ... after transcription and VTT generation
    
    # Cleanup transcription variables
    del result, segments, short_segments, vtt_output
    
    # Unload the model if no longer needed
    del model

    try:
        os.remove(file)
        print(f"File {file} has been deleted")
    except FileNotFoundError:
        print(f"File {file} does not exist")

    print("\n-----FINISHED------");
            

# Connect the button click event to the download function
download_button.on_click(download_file)

# Display the widgets
display(widgets.HBox([url_input, download_button]))
display(output)
