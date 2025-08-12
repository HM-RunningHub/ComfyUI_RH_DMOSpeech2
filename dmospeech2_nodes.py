from .src.infer import DMOInference
import torchaudio
import time
import folder_paths
import os
import soundfile as sf
import uuid
import torch

class RH_DMOSpeech2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_audio": ("AUDIO", ),
                "gen_text": ("STRING", {"multiline": True,
                                      "default": "猫猫真是太帅了，帅的我愿意每个月给他五十万块钱"}),
                "seed": ("INT", {"default": 20, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "The random seed used for creating the noise."}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True,}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "sample"

    CATEGORY = "Runninghub/DMOSpeech2"

    def sample(self, **kwargs):
        
        base_path = os.path.join(folder_paths.models_dir, 'DMOSpeech2', 'ckpts')
        input_audio = kwargs.get('ref_audio')
        ref_audio = os.path.join(folder_paths.get_temp_directory(),  f'dmospeech2_{uuid.uuid4().hex[:8]}.wav')
        torchaudio.save(ref_audio, input_audio["waveform"].squeeze(0), input_audio["sample_rate"])

        # Initialize the model
        tts = DMOInference(
            student_checkpoint_path=os.path.join(base_path, 'model_85000.pt'), 
            duration_predictor_path=os.path.join(base_path, 'model_1500.pt'),
            device="cuda",
            model_type="F5TTS_Base"
        )

        gen_text = kwargs.get('gen_text')
        ref_text = kwargs.get('ref_text')
        if ref_text.strip() == '':
            ref_text = None

        start_time = time.time()
        # Generate with default settings
        generated_audio = tts.generate(
            gen_text=gen_text,
            audio_path=ref_audio,
            prompt_text=ref_text
        )
        end_time = time.time()

        processing_time = end_time - start_time
        audio_duration = generated_audio.shape[-1] / 24000
        rtf = processing_time / audio_duration

        print(f"  RTF: {rtf:.2f}x ({1/rtf:.2f}x speed)")
        print(f"  Processing: {processing_time:.2f}s for {audio_duration:.2f}s audio")
        output_audio_path = os.path.join(folder_paths.get_temp_directory(),  f'dmospeech2_output_{uuid.uuid4().hex[:8]}.wav')
        sf.write(output_audio_path, generated_audio, 24000)
        waveform, sample_rate = torchaudio.load(output_audio_path)
        output_audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (output_audio, )
    
NODE_CLASS_MAPPINGS = {
    "RunningHub DMOSpeech2": RH_DMOSpeech2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub DMOSpeech2": "RunningHub DMOSpeech2",
} 