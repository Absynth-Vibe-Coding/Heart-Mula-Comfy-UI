import os
import torch
import torchaudio
import soundfile as sf
import tempfile
import folder_paths

import comfy.model_management


def _torchaudio_save_soundfile(filepath, src, sample_rate, **kwargs):
    """Wrapper to force soundfile backend for torchaudio.save (avoids torchcodec dependency)."""
    # Use soundfile directly to save audio
    # src shape: [channels, samples] - soundfile expects [samples, channels]
    audio_data = src.numpy().T
    sf.write(filepath, audio_data, sample_rate)


def _torchaudio_load_soundfile(filepath, **kwargs):
    """Wrapper to force soundfile backend for torchaudio.load (avoids torchcodec dependency)."""
    # Use soundfile directly to load audio
    # soundfile returns [samples, channels], torchaudio expects [channels, samples]
    audio_data, sample_rate = sf.read(filepath, dtype='float32')
    # Handle mono audio
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)
    waveform = torch.from_numpy(audio_data.T)
    return waveform, sample_rate


# Monkey-patch torchaudio.save and torchaudio.load to avoid torchcodec requirement
_original_torchaudio_save = torchaudio.save
_original_torchaudio_load = torchaudio.load
torchaudio.save = _torchaudio_save_soundfile
torchaudio.load = _torchaudio_load_soundfile

# Register heartmula models folder
HEARTMULA_MODELS_DIR = os.path.join(folder_paths.models_dir, "heartmula")
os.makedirs(HEARTMULA_MODELS_DIR, exist_ok=True)

# Genre presets: display name -> tags (comma-separated, NO spaces after commas)
GENRE_TAGS = {
    "EDM - Trance": "trance,electronic,synthesizer,energetic,driving,euphoric",
    "EDM - House": "house,electronic,synthesizer,drum machine,groovy,dance",
    "EDM - Techno": "techno,electronic,synthesizer,drum machine,dark,hypnotic",
    "EDM - Dubstep": "dubstep,electronic,synthesizer,bass,heavy,aggressive",
    "Pop - Upbeat": "pop,upbeat,energetic,synthesizer,drums,catchy",
    "Pop - Ballad": "pop,ballad,emotional,piano,strings,romantic",
    "Rock - Alternative": "rock,alternative,electric guitar,drums,energetic",
    "Rock - Indie": "rock,indie,guitar,drums,emotional,melodic",
    "Metal": "metal,heavy,electric guitar,drums,powerful,aggressive",
    "Hip Hop - Trap": "hip hop,trap,drum machine,bass,808,dark",
    "Hip Hop - Boom Bap": "hip hop,boom bap,drums,sampling,groovy,old school",
    "R&B": "r&b,soul,keyboard,smooth,romantic,groovy",
    "Jazz": "jazz,smooth,piano,saxophone,relaxing,sophisticated",
    "Classical": "classical,orchestral,strings,piano,epic,cinematic",
    "Ambient": "ambient,chill,synthesizer,soft,peaceful,atmospheric",
    "Lo-Fi": "lo-fi,chill,relaxing,piano,soft,nostalgic",
    "Country": "country,acoustic,acoustic guitar,warm,heartfelt",
    "Reggae": "reggae,chill,guitar,drums,relaxing,tropical",
    "Funk": "funk,groovy,bass,drums,energetic,rhythmic",
    "Disco": "disco,retro,synthesizer,drums,groovy,dance",
    "Synthwave": "synthwave,retro,80s,synthesizer,electronic,nostalgic",
    "custom": "",
}

GENRE_PRESETS = list(GENRE_TAGS.keys())


def get_heartmula_models():
    """Get list of available HeartMuLa models."""
    models = []
    if os.path.exists(HEARTMULA_MODELS_DIR):
        for name in os.listdir(HEARTMULA_MODELS_DIR):
            model_path = os.path.join(HEARTMULA_MODELS_DIR, name)
            if os.path.isdir(model_path):
                # Check for HeartMuLa-oss-3B or HeartMuLa-oss-7B subfolder
                has_3b = os.path.exists(os.path.join(model_path, "HeartMuLa-oss-3B", "config.json"))
                has_7b = os.path.exists(os.path.join(model_path, "HeartMuLa-oss-7B", "config.json"))
                if has_3b or has_7b:
                    models.append(name)
    return models if models else ["no models found"]


class HeartMuLaLoader:
    """Load the HeartMuLa model."""

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_heartmula_models(), {
                    "tooltip": "Select a HeartMuLa model from models/heartmula folder"
                }),
                "version": (["3B", "7B"], {
                    "default": "3B",
                    "tooltip": "Model version to load"
                }),
            }
        }

    RETURN_TYPES = ("HEARTMULA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/heartmula"
    DESCRIPTION = "Load the HeartMuLa music generation model from models/heartmula folder."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force refresh of model list
        return float("nan")

    def load_model(self, model_name: str, version: str):
        if model_name == "no models found":
            raise ValueError(f"No HeartMuLa models found in {HEARTMULA_MODELS_DIR}. Please download a model.")

        if model_name == ".":
            model_path = HEARTMULA_MODELS_DIR
        else:
            model_path = os.path.join(HEARTMULA_MODELS_DIR, model_name)

        cache_key = f"{model_path}_{version}"

        if cache_key in self._model_cache:
            return (self._model_cache[cache_key],)

        # Free up VRAM before loading large model
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        from heartlib import HeartMuLaGenPipeline

        device = comfy.model_management.get_torch_device()

        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            version=version,
            dtype=torch.bfloat16,
            device=device
        )

        self._model_cache[cache_key] = pipe
        return (pipe,)


class HeartMuLaGenerate:
    """Generate music using HeartMuLa from lyrics and tags."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HEARTMULA_MODEL",),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[verse]\nYour lyrics here\n\n[chorus]\nChorus lyrics here",
                    "tooltip": "Lyrics with section markers like [verse], [chorus], etc."
                }),
                "genre": (GENRE_PRESETS, {
                    "default": "edm, trance",
                    "tooltip": "Select a genre preset or 'custom' to use custom_tags"
                }),
                "custom_tags": ("STRING", {
                    "default": "",
                    "tooltip": "Custom tags (only used when genre is 'custom')"
                }),
            },
            "optional": {
                "max_duration_sec": ("INT", {
                    "default": 40,
                    "min": 5,
                    "max": 240,
                    "step": 5,
                    "tooltip": "Max audio duration in seconds. Steps: 5s=63, 10s=125, 20s=250, 30s=375, 60s=750, 120s=1500 (1 step = 80ms)"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature (higher = more random)"
                }),
                "topk": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500,
                    "tooltip": "Top-k sampling parameter"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "codec_steps": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Codec decoding steps (lower=faster, 5-10 recommended). 10 steps ~3min, 5 steps ~1.5min"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Random seed (0 for random)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/heartmula"
    DESCRIPTION = "Generate music from lyrics and style tags using HeartMuLa."

    def generate(
        self,
        model,
        lyrics: str,
        genre: str,
        custom_tags: str = "",
        max_duration_sec: int = 20,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: float = 1.0,
        codec_steps: int = 5,
        seed: int = 0,
    ):
        if seed != 0:
            torch.manual_seed(seed)

        # Use custom_tags if genre is "custom", otherwise look up tags from preset
        tags = custom_tags if genre == "custom" else GENRE_TAGS.get(genre, genre)
        print(f"[HeartMuLa] Using tags: {tags}")

        # Create temp files for lyrics and tags (UTF-8 encoding for Windows compatibility)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(lyrics)
            lyrics_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(tags)
            tags_path = f.name

        # Output path (WAV to avoid FFmpeg dependency)
        output_path = os.path.join(tempfile.gettempdir(), "heartmula_output.wav")

        # Convert duration to ms
        max_audio_length_ms = max_duration_sec * 1000
        steps = max_audio_length_ms // 80
        print(f"[HeartMuLa] max_duration_sec received: {max_duration_sec}")
        print(f"[HeartMuLa] Generating {max_duration_sec}s audio ({steps} steps, {max_audio_length_ms}ms)")

        # Patch tqdm to support cancellation
        import heartlib.pipelines.music_generation as hm_gen
        from tqdm import tqdm as original_tqdm

        class InterruptibleTqdm(original_tqdm):
            def __iter__(self):
                for item in super().__iter__():
                    if comfy.model_management.processing_interrupted():
                        print("[HeartMuLa] Generation cancelled by user")
                        self.close()
                        raise InterruptedError("Generation cancelled")
                    yield item

        # Temporarily replace tqdm in heartlib
        hm_gen.tqdm = InterruptibleTqdm

        try:
            # Run generation - inputs dict separate from kwargs
            print(f"[HeartMuLa] Using codec_steps={codec_steps}")
            model(
                {"lyrics": lyrics_path, "tags": tags_path},
                save_path=output_path,
                max_audio_length_ms=max_audio_length_ms,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
                codec_steps=codec_steps,
            )

            # Load the generated audio
            print("[HeartMuLa] Loading generated audio...")
            waveform, sample_rate = torchaudio.load(output_path)

            # Add batch dimension: [Channels, Samples] -> [Batch, Channels, Samples]
            waveform = waveform.unsqueeze(0)
            print(f"[HeartMuLa] Audio ready: {waveform.shape[2] / sample_rate:.1f}s @ {sample_rate}Hz")

            audio_output = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }

            return (audio_output,)

        except InterruptedError:
            # User cancelled - return empty/silent audio
            print("[HeartMuLa] Returning empty audio due to cancellation")
            waveform = torch.zeros(1, 2, 44100)  # 1 second of silence
            return ({"waveform": waveform, "sample_rate": 44100},)

        finally:
            # Restore original tqdm
            hm_gen.tqdm = original_tqdm
            # Cleanup temp files
            for path in [lyrics_path, tags_path]:
                if os.path.exists(path):
                    os.remove(path)


class HeartMuLaGenerateFromFiles:
    """Generate music using HeartMuLa from lyrics and tags files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HEARTMULA_MODEL",),
                "lyrics_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to lyrics .txt file"
                }),
                "genre": (GENRE_PRESETS, {
                    "default": "edm, trance",
                    "tooltip": "Select a genre preset or 'custom' to use tags_file"
                }),
                "tags_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to tags .txt file (only used when genre is 'custom')"
                }),
            },
            "optional": {
                "max_duration_sec": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 240,
                    "step": 10,
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "topk": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500,
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "codec_steps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Codec decoding steps (lower=faster, 5-10 recommended)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/heartmula"
    DESCRIPTION = "Generate music from lyrics and tags files using HeartMuLa."

    def generate(
        self,
        model,
        lyrics_file: str,
        genre: str,
        tags_file: str = "",
        max_duration_sec: int = 60,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: float = 1.0,
        codec_steps: int = 5,
        seed: int = 0,
    ):
        if seed != 0:
            torch.manual_seed(seed)

        output_path = os.path.join(tempfile.gettempdir(), "heartmula_output.wav")

        # Use tags_file if genre is "custom", otherwise create temp file with genre preset
        if genre == "custom":
            actual_tags_file = tags_file
        else:
            tags = GENRE_TAGS.get(genre, genre)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(tags)
                actual_tags_file = f.name

        model(
            {"lyrics": lyrics_file, "tags": actual_tags_file},
            save_path=output_path,
            max_audio_length_ms=max_duration_sec * 1000,
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale,
            codec_steps=codec_steps,
        )

        # Cleanup temp tags file if we created one
        if genre != "custom" and os.path.exists(actual_tags_file):
            os.remove(actual_tags_file)

        waveform, sample_rate = torchaudio.load(output_path)
        waveform = waveform.unsqueeze(0)

        return ({
            "waveform": waveform,
            "sample_rate": sample_rate,
        },)


NODE_CLASS_MAPPINGS = {
    "HeartMuLaLoader": HeartMuLaLoader,
    "HeartMuLaGenerate": HeartMuLaGenerate,
    "HeartMuLaGenerateFromFiles": HeartMuLaGenerateFromFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLaLoader": "HeartMula Absynth Music Generator - Load Model",
    "HeartMuLaGenerate": "HeartMula Absynth Music Generator",
    "HeartMuLaGenerateFromFiles": "HeartMula Absynth Music Generator (Files)",
}
