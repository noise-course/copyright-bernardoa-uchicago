import librosa
import chromaprint
from pydub import AudioSegment
import numpy as np
from scipy.spatial.distance import cosine
import os
"""
################################### DISCRETION ####################################

1. The functions average_mfcc and cosine_similarity_fast, were entirely written by ChatGPT but revised by me.

2. The other functions were of my own personal creation with some syntax corrections from ChatGPT

3. All other code was entirely written by me.

###################################################################################
"""
folder = "/Users/bernardoalbuquerque/Security/temp_Copyright/music"
files = os.listdir(folder)
sorted = sorted(files)
og = []
for file in sorted:
    if "shifted" not in file and "ms" not in file:
        og.append(file)

def pitch_shift(files):
    for filename in files:
        input_path = os.path.join(folder, filename)
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(folder, f"{base}_shifted.mp3")

        y, sr = librosa.load(input_path)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)

        audio = AudioSegment(
            (y_shifted * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        audio.export(output_path, format="mp3")
        print("Saved:", output_path)
        return
    
def snippets(files):
    os.chdir("music")
    for filename in files:
        audio = AudioSegment.from_mp3(filename)
        base = os.path.splitext(filename)[0]

        start_ms = 0   
        end_ms   = 5 * 1000

        snippet = audio[start_ms:end_ms]
        snippet.export(f"{base}_5ms.mp3", format="mp3")
        print("Saved snippet")

        start_ms = 0   
        end_ms = 5 * 1000

        snippet = audio[start_ms:end_ms]
        snippet.export(f"{base}_5ms.mp3", format="mp3")
        print("Saved snippet")

        start_ms = 0   
        end_ms = 15 * 1000

        snippet = audio[start_ms:end_ms]
        snippet.export(f"{base}_15ms.mp3", format="mp3")
        print("Saved snippet")

        start_ms = 0   
        end_ms = 30 * 1000

        snippet = audio[start_ms:end_ms]
        snippet.export(f"{base}_30ms.mp3", format="mp3")
        print("Saved snippet")
        return

def make_fingerprints(files):
    fingerprints = []
    for filename in files:
        audio = AudioSegment.from_mp3(f"/Users/bernardoalbuquerque/Security/temp_Copyright/music/{filename}")
        samples = np.array(audio.get_array_of_samples())
        
        channels = audio.channels
        sample_rate = audio.frame_rate
        if channels > 1:
            samples = samples.reshape((-1, channels))[:,0]

        fp_ctx = chromaprint.Fingerprinter()
        fp_ctx.start(sample_rate, 1)
        fp_ctx.feed(samples.tobytes())
        fingerprint = fp_ctx.finish()

        fingerprints.append({
            "filename": filename,
            "fingerprint": fingerprint,
        })
        print(f"Fingerprinted: {filename}")
    return fingerprints

def fingerprint_similarity(fp1, fp2):
    int_fp1, _ = chromaprint.decode_fingerprint(fp1)
    int_fp2, _ = chromaprint.decode_fingerprint(fp2)

    int_fp1 = np.array(int_fp1).flatten()
    int_fp2 = np.array(int_fp2).flatten()
    min_len = min(len(int_fp1), len(int_fp2))
    int_fp1 = int_fp1[:min_len]
    int_fp2 = int_fp2[:min_len]

    # Hamming distance
    dist = np.sum(int_fp1 != int_fp2) / min_len
    similarity = 1 - dist
    return similarity

def chroma_similarity(file1, file2):
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)
    
    C1 = librosa.feature.chroma_stft(y=y1, sr=sr1)
    C2 = librosa.feature.chroma_stft(y=y2, sr=sr2)
    
    vec1 = np.mean(C1, axis=1)
    vec2 = np.mean(C2, axis=1)
    
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim

def average_mfcc(filename, sr=22050, n_mfcc=13, hop_length=512):
    y, _ = librosa.load(filename, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return np.mean(mfcc, axis=1)

def cosine_similarity_fast(song1_file, song2_file):
    vec1 = average_mfcc(song1_file)
    vec2 = average_mfcc(song2_file)
    return 1 - cosine(vec1, vec2)


#pitch_shift(sorted)
#snippets(og)
os.chdir("music")
limit = 0.99
tp = 0
fp = 0
fn = 0
fp_list = []
fn_list = []
for i in range(len(sorted)-1):
    for j in range(i+1, len(sorted)):
        if int(sorted[i].split(" ")[0]) == int(sorted[j].split(" ")[0]):
            if cosine_similarity_fast(sorted[i], sorted[j]) >= limit:
                tp += 1
            else:
                fn += 1
                fn_list.append(f"{sorted[i]} vs {sorted[j]}")
        elif int(sorted[i].split(" ")[0]) != int(sorted[j].split(" ")[0]):
            if cosine_similarity_fast(sorted[i], sorted[j]) < limit:
                tp += 1
            else:
                fp += 1
                fp_list.append(f"{sorted[i]} vs {sorted[j]}")
    print(f"Done with track #{i+1}")
print("TP:", tp) #1959
print("FP:", fp) #366
print("FN:", fn) #90

ms_ms = 0
ms_shifted = 0
ms_blank = 0
shifted_ms = 0
shifted_shifted = 0
shifted_blank = 0
blank_ms = 0
blank_shifted = 0
blank_blank = 0
for items in fn_list:
    first = items.split(" vs ")[0]
    second = items.split(" vs ")[1]
    if "ms" in first and "ms" in second:
        ms_ms += 1
    elif "ms" in first and "shifted" in second:
        ms_shifted += 1
    elif "ms" in first and "shifted" not in second and "ms" not in second:
        ms_blank += 1
    elif "shifted" in first and "ms" in second:
        shifted_ms += 1
    elif "shifted" in first and "shifted" in second:
        shifted_shifted += 1
    elif "shifted" in first and "shifted" not in second and "ms" not in second:
        shifted_blank += 1
    elif "shifted" not in first and "ms" not in first and "ms" in second:
        blank_ms += 1
    elif "shifted" not in first and "ms" not in first and "shifted" in second:
        blank_shifted += 1
    elif "shifted" not in first and "ms" not in first and "ms" not in second and "shifted" not in second:
        blank_blank += 1
print("Flase Negative Results")
print("ms_ms:", ms_ms)
print("ms_shifted:", ms_shifted)
print("ms_blank:", ms_blank)
print("shifted_ms:", shifted_ms)
print("shifted_shifted:", shifted_shifted)
print("shifted_blank:", shifted_blank)
print("blank_ms:", blank_ms)
print("blank_shifted:", blank_shifted)
print("blank_blank:", blank_blank)
print()

ms_ms = 0
ms_shifted = 0
ms_blank = 0
shifted_ms = 0
shifted_shifted = 0
shifted_blank = 0
blank_ms = 0
blank_shifted = 0
blank_blank = 0
for items in fp_list:
    first = items.split(" vs ")[0]
    second = items.split(" vs ")[1]
    if "ms" in first and "ms" in second:
        ms_ms += 1
    elif "ms" in first and "shifted" in second:
        ms_shifted += 1
    elif "ms" in first and "shifted" not in second and "ms" not in second:
        ms_blank += 1
    elif "shifted" in first and "ms" in second:
        shifted_ms += 1
    elif "shifted" in first and "shifted" in second:
        shifted_shifted += 1
    elif "shifted" in first and "shifted" not in second and "ms" not in second:
        shifted_blank += 1
    elif "shifted" not in first and "ms" not in first and "ms" in second:
        blank_ms += 1
    elif "shifted" not in first and "ms" not in first and "shifted" in second:
        blank_shifted += 1
    elif "shifted" not in first and "ms" not in first and "ms" not in second and "shifted" not in second:
        blank_blank += 1
print("False Positive results")
print("ms_ms:", ms_ms)
print("ms_shifted:", ms_shifted)
print("ms_blank:", ms_blank)
print("shifted_ms:", shifted_ms)
print("shifted_shifted:", shifted_shifted)
print("shifted_blank:", shifted_blank)
print("blank_ms:", blank_ms)
print("blank_shifted:", blank_shifted)
print("blank_blank:", blank_blank)
print()
