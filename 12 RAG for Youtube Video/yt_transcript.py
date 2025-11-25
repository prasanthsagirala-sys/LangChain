from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import os 

def get_yt_video_en_transcript(video_id):
    #video_id = "Gfr50f6ZBvo"  # only the ID, not full URL
    filename = video_id + ".txt"

    # 1️⃣ Check if transcript file already exists
    if os.path.exists(filename):
        print("Transcript file already exists. Loading from disk...")
        with open(filename, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        print("Transcript file not found. Fetching from YouTube...")
        try:
            # Your original fetching logic
            transcript_list = YouTubeTranscriptApi().list(video_id)
            transcript_en = transcript_list.find_transcript(['en'])
            transcript_chunks = transcript_en.fetch()
            transcript = ' '.join(i.text for i in transcript_chunks)

            # Save to file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(transcript)

            print("Transcript fetched and saved to", filename)

        except TranscriptsDisabled:
            print("No captions available for this video (transcripts disabled).")
            transcript = None
        except NoTranscriptFound:
            print("No transcript found for this video.")
            transcript = None
        except Exception as e:
            print("Error while fetching transcript:", e)
            transcript = None

    # At this point, `transcript` is either the loaded/fetched text or None
    # You can add a check before using it:
    if transcript:
        print("Transcript length:", len(transcript))
    
    return transcript

