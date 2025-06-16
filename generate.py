# generate.py
"""
セマンティックバージョニング表記。更新時に ver を必ず更新すること！！
"""
__version__ = "0.15.12"  # メジャー=0（開発中）、マイナー=15（機能追加回数）、パッチ=12（デバッグ回数）

import os
import glob
import random
import numpy as np
import pretty_midi
import tensorflow as tf
from train import build_model
from train_data import build_dataset

def sample_pitch(logits, temp=1.0):
    probs = tf.nn.softmax(logits / temp).numpy()
    return np.random.choice(len(probs), p=probs)

def append_event(events, pitch, start, duration, vel_range=(90,120)):
    vel = random.randint(*vel_range)
    end = start + duration
    events.append((int(pitch), start, end, vel))
    return end
def generate_section(model, seq, n_notes, temp, scale_set, durations):
    events = []
    start = 0.0
    for _ in range(n_notes):
        inp = seq[np.newaxis, ...]
        preds = model.predict(inp, verbose=0)
        pitch = sample_pitch(preds['pitch'][0], temp)
        # Fマイナー音階にスナップ(近いうちにGbに変えるか、ランダムにしたい。(願望)
        if pitch % 12 not in scale_set:
            allowed = [n for n in range(128) if n % 12 in scale_set]
            pitch = min(allowed, key=lambda x: abs(x - pitch))
        # ノート長は必ず 1/4 または 1/8
        duration = random.choice(durations)
        start = append_event(events, pitch, start, duration)
        # シード更新（pitch, dummy_step, dummy_dur はコードの互換性維持用だけど多分すぐ消すのかもわからん）
        step = 0.0
        new_ev = np.array([pitch, step, duration], dtype=np.float32)
        seq = np.vstack([seq[1:], new_ev])
    return events, seq

def main():
    #  ランドォーム
    random.seed(None)
    np.random.seed(None)
    tf.random.set_seed(None)

    # === 設定 ===
    bpm = 120
    seq_length = 20
    output_dir = 'outputs'
    model_weights = 'models/best_model.weights.h5'
    #（MIDI mod12）
    f_minor = {5, 7, 8, 10, 0, 1, 3}
    seconds_per_beat = 60.0 / bpm
    quarter = seconds_per_beat       # 1/4 ノート
    eighth  = seconds_per_beat / 2   # 1/8 ノート
    durations = [quarter, eighth]
    model = build_model(seq_length)
    model.load_weights(model_weights)
    X, _, _, _ = build_dataset(data_dir='data', seq_length=seq_length)
    seq = random.choice(X).copy()

    all_events = []
    current_time = 0.0
    
    # 1) リフ部
    riff_events, seq = generate_section(
        model, seq,
        n_notes=8,
        temp=0.8,
        scale_set=f_minor,
        durations=durations
    )
    for p, s, e, v in riff_events:
        all_events.append((p, s + current_time, e + current_time, v))
    current_time += sum(e - s for _, s, e, _ in riff_events)

    # 2) ビルドアップ
    bars_up = random.randint(4, 6)
    notes_up = bars_up * 4
    bu_events, seq = generate_section(
        model, seq,
        n_notes=notes_up,
        temp=1.0,
        scale_set=f_minor,
        durations=durations
    )
    for p, s, e, v in bu_events:
        all_events.append((p, s + current_time, e + current_time, v))
    current_time += sum(e - s for _, s, e, _ in bu_events)

    # 3) サビ部
    bars_chorus = random.randint(8, 16)
    notes_chorus = bars_chorus * 4
    chorus_events, seq = generate_section(
        model, seq,
        n_notes=notes_chorus,
        temp=1.3,
        scale_set=f_minor,
        durations=durations
    )
    for p, s, e, v in chorus_events:
        all_events.append((p, s + current_time, e + current_time, v))
    current_time += sum(e - s for _, s, e, _ in chorus_events)

    # 4) ブレイク
    current_time += 2 * 4 * seconds_per_beat

    # 5) アウトロ
    outro_events, _ = generate_section(
        model, seq,
        n_notes=8,
        temp=1.0,
        scale_set=f_minor,
        durations=durations
    )
    for p, s, e, v in outro_events:
        all_events.append((p, s + current_time, e + current_time, v))

    # =生成==
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(f"{output_dir}/{__version__}_*.mid")
    idx = max([int(os.path.splitext(f)[0].split('_')[-1]) for f in existing] + [0]) + 1
    out_path = f"{output_dir}/{__version__}_{idx}.mid"

    # MIDI 書き出し
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=80)
    for p, s, e, v in all_events:
        inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))
    pm.instruments.append(inst)
    pm.write(out_path)

    print(f"Generated EDM melody (BPM={bpm}, ver={__version__}): {out_path}")

if __name__ == '__main__':
    main()
