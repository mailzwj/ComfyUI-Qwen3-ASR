# ComfyUI-Qwen3-ASR Node Implementation
# Based on the open-source Qwen3-ASR project by Alibaba Qwen team

import os
import torch
from typing import Dict, Any, Tuple
import folder_paths


# Common languages list for UI：
# 中文 (zh)、英文 (en)、粤语 (yue)、阿拉伯语 (ar)、德语 (de)、法语 (fr)、西班牙语 (es)、葡萄牙语 (pt)、印尼语 (id)、意大利语 (it)、
# 韩语 (ko)、俄语 (ru)、泰语 (th)、越南语 (vi)、日语 (ja)、土耳其语 (tr)、印地语 (hi)、马来语 (ms)、荷兰语 (nl)、瑞典语 (sv)、丹麦语 (da)、芬兰语 (fi)、
# 波兰语 (pl)、捷克语 (cs)、菲律宾语 (fil)、波斯语 (fa)、希腊语 (el)、匈牙利语 (hu)、马其顿语 (mk)、罗马尼亚语 (ro)
DEMO_LANGUAGES = [
    "Auto",
    "中文",
    "英文",
    "粤语",
    "阿拉伯语",
    "德语",
    "法语",
    "西班牙语",
    "葡萄牙语"
    "印尼语",
    "意大利语",
    "韩语",
    "俄语",
    "泰语",
    "越南语",
    "日语",
    "土耳其语",
    "印地语",
    "马来语",
    "荷兰语",
    "瑞典语",
    "丹麦语",
    "芬兰语",
    "波兰语",
    "捷克语",
    "菲律宾语",
    "波斯语",
    "希腊语",
    "匈牙利语",
    "马其顿语",
    "罗马尼亚语",
]

# Language mapping dictionary to engine codes
#  ['Chinese', 'English', 'Cantonese', 'Arabic', 'German', 'French', 'Spanish', 'Portuguese', 'Indonesian', 'Italian', 'Italian', 'Russian', 'Thai', 'Vietnamese', 'Japanese',
# 'Turkish', 'Hindi', 'Malay', 'Dutch', 'Swedish', 'Danish', 'Finnish', 'Polish', 'Czech', 'Filipino', 'Persian', 'Greek', 'Romanian', 'Hungarian', 'Macedonian']
LANGUAGE_MAP = {
    "Auto": "auto",
    "中文": "Chinese",
    "英文": "English",
    "粤语": "Cantonese",
    "阿拉伯语": "Arabic",
    "德语": "German",
    "法语": "French",
    "西班牙语": "Spanish",
    "葡萄牙语": "Portuguese",
    "印尼语": "Indonesian",
    "意大利语": "Italian",
    "韩语": "Italian",
    "俄语": "Russian",
    "泰语": "Thai",
    "越南语": "Vietnamese",
    "日语": "Japanese",
    "土耳其语": "Turkish",
    "印地语": "Hindi",
    "马来语": "Malay",
    "荷兰语": "Dutch",
    "瑞典语": "Swedish",
    "丹麦语": "Danish",
    "芬兰语": "Finnish",
    "波兰语": "Polish",
    "捷克语": "Czech",
    "菲律宾语": "Filipino",
    "波斯语": "Persian",
    "希腊语": "Greek",
    "匈牙利语": "Hungarian",
    "马其顿语": "Macedonian",
    "罗马尼亚语": "Romanian",
}

# Language for aligner (if used)
# 中文、英文、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语
ALIGNER_LANGUAGES = ["中文", "英文", "粤语", "法语", "德语", "意大利语", "日语", "韩语", "葡萄牙语", "俄语", "西班牙语"]

# Model family options for UI (0.6B / 1.7B)
MODEL_FAMILIES = ["0.6B", "1.7B"]
# Mapping of family to default HuggingFace repo ID
MODEL_FAMILY_TO_HF = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}

# All required models for batch download
ALL_MODELS = [
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B"
]

_MODELS_CHECKED = False

try:
    # 1. Try absolute import first (if user installed via pip)
    import qwen_asr
    Qwen3ASRModel = qwen_asr.Qwen3ASRModel
except ImportError:
    try:
        # 2. Fallback to local package import (relative or absolute via sys.path)
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        import traceback
        print(f"\n❌ [Qwen3-ASR] Critical Import Error: {e}")
        print("   Traceback for debugging:")
        traceback.print_exc()
        print("\n   Common fix: run 'pip install -r requirements.txt' in your ComfyUI environment.")
        
        Qwen3ASRModel = None


# Global model cache
_MODEL_CACHE = {}

def download_model_if_needed(model_id: str, qwen_root: str) -> str:
    """Download a specific model if not found locally"""
    folder_name = model_id.split("/")[-1]
    target_dir = os.path.join(qwen_root, folder_name)
    
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Model already exists
        return target_dir
    
    print(f"\n📥 [Qwen3-ASR] Downloading {model_id}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=target_dir)
        print(f"✅ [Qwen3-ASR] {folder_name} downloaded successfully.\n")
        return target_dir
    except ImportError:
        print("⚠️ [Qwen3-ASR] 'huggingface_hub' not found. Please install it to use auto-download.")
        return None
    except Exception as e:
        print(f"❌ [Qwen3-ASR] Failed to download {model_id}: {e}")
        return None

def load_qwen_model(model_choice: str, device: str, precision: str, max_inference_batch_size: int, max_new_tokens: int, forced_aligner: bool):
    """Shared model loading logic with caching and local path priority"""
    global _MODEL_CACHE
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # 针对 Mac 的关键修复
        else:
            device = "cpu"
    
    # 强制 Mac 使用 float16 或 bfloat16 (MPS 跑 float32 会很慢)
    if device == "mps" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    # Set precision
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
        
    # Cache key
    cache_key = (model_choice, device, precision)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    # Clear old cache
    _MODEL_CACHE.clear()
    
    # --- 1. Determine search directories ---
    base_paths = []
    try:
        # comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        comfy_root = folder_paths.models_dir
        qwen_asr_dir = os.path.join(comfy_root, "qwen-asr")
        if os.path.exists(qwen_asr_dir):
            base_paths.append(qwen_asr_dir)
    except Exception:
        pass

    # --- 2. Search for matching models ---
    HF_MODEL_MAP = {
        "0.6B": "Qwen/Qwen3-ASR-0.6B",
        "1.7B": "Qwen/Qwen3-ASR-1.7B",
    }
    
    final_source = HF_MODEL_MAP.get(model_choice) or "Qwen/Qwen3-ASR-1.7B"
    found_local = None
    
    for base in base_paths:
        try:
            if not os.path.isdir(base): continue
            subdirs = os.listdir(base)
            for d in subdirs:
                cand = os.path.join(base, d)
                if os.path.isdir(cand):
                    # Match logic: contains model size and type keyword
                    if 'Qwen3-ASR-' + model_choice == d:
                        found_local = cand
                        break
            if found_local: break
        except Exception: pass
    
    if found_local:
        final_source = found_local
        print(f"✅ [Qwen3-ASR] Loading local model: {os.path.basename(final_source)}")
    else:
        # Try to download the specific model if not found locally
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
        comfy_models_path = folder_paths.models_dir
        qwen_root = os.path.join(comfy_models_path, "qwen-asr")
        
        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"✅ [Qwen3-ASR] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            # Fall back to remote loading if download failed
            print(f"🌐 [Qwen3-ASR] Loading remote model: {final_source}")

    if Qwen3ASRModel is None:
        raise RuntimeError(
            "❌ [Qwen3-ASR] Model class is not loaded because the 'qwen_asr' package failed to import. "
            "Please check the ComfyUI console for the detailed 'Critical Import Error' above."
        )
    
    aligner_path = os.path.join(folder_paths.models_dir, "qwen-asr", "Qwen3-ForcedAligner-0.6B")

    # Try to use flash_attention_2 if available, otherwise fall back to default
    try:
        model = Qwen3ASRModel.from_pretrained(
            final_source,
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            forced_aligner=aligner_path if forced_aligner else None,
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device,
                attn_implementation="flash_attention_2",
            ) if forced_aligner else None,
        )
    except (ImportError, ValueError, Exception) as e:
        # flash_attention_2 not available or not supported, use default attention
        print(f"⚠️ [Qwen3-ASR] flash_attention_2 not available, using default attention: {e}")
        model = Qwen3ASRModel.from_pretrained(
            final_source,
            device_map=device,
            dtype=dtype,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            forced_aligner=aligner_path if forced_aligner else None,
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device,
                # attn_implementation="flash_attention_2",
            ) if forced_aligner else None,
        )
    
    _MODEL_CACHE[cache_key] = model
    return model

class Voice2TextNode:
    """
    Voice2TextNode (ASR) Node: 语音转文本
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "max_inference_batch_size": ("INT", {"default": 32}),
                "max_new_tokens": ("INT", {"default": 256}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
                "forced_aligner": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "ANY", "STRING")
    RETURN_NAMES = ("文本", "语言", "time_stamps", "time_stamps_json")
    FUNCTION = "generate"
    CATEGORY = "Qwen3-ASR"
    DESCRIPTION = "Voice2Text: Convert audio to text."

    def generate(self, audio: Dict[str, Any], max_inference_batch_size: int, max_new_tokens: int, model_choice: str, device: str, precision: str, language: str, forced_aligner: bool, seed: int) -> Tuple[str]:
        if not audio or "waveform" not in audio:
            raise RuntimeError("Audio is required")

        # Load model
        model = load_qwen_model(model_choice, device, precision, max_inference_batch_size, max_new_tokens, forced_aligner)

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed % (2**32))

        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        results = model.transcribe(
            audio=(audio.get("waveform").mean(dim=0).numpy(), audio.get("sample_rate")),
            language=None if mapped_lang == "auto" else mapped_lang,
            return_time_stamps=True if forced_aligner else False,
        )

        if results and len(results) > 0:
            rs = results[0]

            ts_json = ""
            if forced_aligner and rs.time_stamps is not None:
                import json
                ts_list = []
                for segment in rs.time_stamps.items:
                    ts_list.append({
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "text": segment.text,
                    })
                ts_json = json.dumps(ts_list, ensure_ascii=False, indent=2)

            return (rs.text, rs.language, rs.time_stamps, ts_json)
        raise RuntimeError("Invalid audio data generated")

