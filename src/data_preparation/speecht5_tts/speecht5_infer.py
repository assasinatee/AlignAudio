import os, json, torch, numpy as np, soundfile as sf
from tqdm import tqdm
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import argparse

# Load pretrained models from Hugging Face Hub
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


def main(args):
    # Speaker embedding paths (x-vectors from CMU Arctic dataset)
    spk_emb = {
        "BDL": "./src/data_preparation/speecht5_tts/spkemd/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
        "CLB": "./src/data_preparation/speecht5_tts/spkemd/cmu_us_clb_arctic-wav-arctic_a0144.npy",
        "KSP": "./src/data_preparation/speecht5_tts/spkemd/cmu_us_ksp_arctic-wav-arctic_b0087.npy",
        "RMS": "./src/data_preparation/speecht5_tts/spkemd/cmu_us_rms_arctic-wav-arctic_b0353.npy",
        "SLT": "./src/data_preparation/speecht5_tts/spkemd/cmu_us_slt_arctic-wav-arctic_a0508.npy",
    }
    # Preload and convert to tensors for efficiency
    spk_emb = {k: torch.tensor(np.load(v)).unsqueeze(0)
               for k, v in spk_emb.items()}

    # Parse multiple split files (comma-separated or list)
    split_files = args.split_files.split(',') if isinstance(args.split_files, str) and ',' in args.split_files else [args.split_files]

    for split_file in split_files:
        # Extract split name from path (e.g., 'test', 'val', 'train')
        split = split_file.split('/')[-2]
        
        # Construct output directory: {save_dir}/{speaker}/{split}
        save_dir = os.path.join(args.save_dir, args.speaker, split)
        os.makedirs(save_dir, exist_ok=True)

        # Load audio metadata with captions
        data = json.load(open(split_file))['audios']
        
        # Generate speech for each audio item
        for item in tqdm(data, desc=f"Processing {split}"):
            wav_path = os.path.join(save_dir, f"{item['audio_id']}.wav")
            if os.path.exists(wav_path):
                continue  # Skip if already generated

            # Get the first caption text
            text = item["captions"][0]["caption"]
            
            # Tokenize and truncate to max length
            inputs = processor(text=text, return_tensors="pt")
            input_ids = inputs["input_ids"][:, :model.config.max_text_positions]
            input_ids = torch.tensor(input_ids)

            # Generate speech with speaker conditioning
            with torch.no_grad():
                speech = model.generate(
                    input_ids=input_ids,
                    speaker_embeddings=spk_emb[args.speaker],
                    vocoder=vocoder
                ).squeeze()

            # Save as 16kHz WAV file
            sf.write(wav_path, speech.cpu().float().numpy(), 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeechT5 TTS Batch Generation")
    parser.add_argument("--speaker", type=str, default="RMS", 
                        choices=["BDL", "CLB", "KSP", "RMS", "SLT"],
                        help="Speaker ID (CMU Arctic speaker)")
    parser.add_argument("--split_files", type=str, required=True,
                        help="Comma-separated JSON file paths containing captions")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Root directory for output WAV files")
    
    args = parser.parse_args()
    main(args)