import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

def stitch_videos(input_dir='videos', output_file='stitched_video.mp4', extensions=['.mp4', '.avi', '.mov']):
    """Stitch all video files in the input directory into a single video.

    Args:
        input_dir (str): Directory containing the video files.
        output_file (str): Path to the output stitched video.
        extensions (list): List of video file extensions to include.

    Returns:
        None
    """
    # Get list of video files in the directory
    video_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(tuple(extensions))
    ])

    if not video_files:
        print(f"No video files found in {input_dir} with extensions {extensions}.")
        return

    print(f"Found {len(video_files)} video files: {video_files}")

    # Load each video as a clip
    clips = []
    for video_path in video_files:
        try:
            clip = VideoFileClip(video_path)
            clips.append(clip)
            print(f"Loaded {video_path}")
        except Exception as e:
            print(f"Error loading {video_path}: {e}")

    if not clips:
        print("No valid clips loaded. Exiting.")
        return

    # Concatenate all clips into one
    final_clip = concatenate_videoclips(clips, method="compose")

    # Write the output video
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
    print(f"Stitched video saved to {output_file}")

# Example usage: Run this in your script or adjust parameters as needed
if __name__ == "__main__":
    stitch_videos(input_dir='/workspaces/PhoenX_RL/src/app/assets/videos', output_file='output_stitched.mp4')