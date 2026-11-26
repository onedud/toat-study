"""Tody ASMR 4종 빌드 (v49)

재현 방법
  1) 아래 소스를 Wikimedia Commons에서 받아 ./src 에 두고
  2) ffmpeg(imageio-ffmpeg)로 44.1kHz 스테레오 wav → work_<이름>.wav 로 디코딩한 뒤
  3) python3 tools/asmr-build.py  →  out/{library,rain,cafe,kitchen}.mp3

 — 실제 녹음 → 42초 크로스페이드 루프 → 잔잔한 음량 → MP3.

소스(모두 Wikimedia Commons, PD/CC0/CC BY):
  rain     : File:Rain (1).ogg — ezwa, Public domain
  cafe     : File:Cafe ambiance.ogg — Marble Toast, CC0  + Restaurant ambience.ogg (stephan, PD) 식기 레이어
  library  : File:20121112 TU Delft Library, quiet study room ... — el_mar, CC BY 3.0
             + File:Turning a page.ogg / Old book.ogg (PD) 책장 넘기는 소리
  kitchen  : File:Ambient Kitchen Sounds.wav — CC0
"""
import json, os, subprocess, sys
import numpy as np
import soundfile as sf

FF = subprocess.check_output([sys.executable, '-c', 'import imageio_ffmpeg as f;print(f.get_ffmpeg_exe())']).decode().strip()
SR = 44100
L = 42.0          # 루프 길이(초)
F = 3.0           # 이음매 크로스페이드(초)
TARGET_LUFS = -29.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)


def load(path, start, dur, ch=None):
    f = sf.SoundFile(path)
    f.seek(int(start * f.samplerate))
    x = f.read(int(dur * f.samplerate), dtype='float32', always_2d=True)
    f.close()
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x


def resp_filter(x, kind, freq, gain_db=0.0, order=2):
    """영상(零相位) FFT 필터 — 잡음 소재라 위상은 문제되지 않는다."""
    n = len(x)
    pad = 1 << 15
    xp = np.pad(x, ((pad, pad), (0, 0)), mode='reflect')
    N = len(xp)
    fr = np.fft.rfftfreq(N, 1 / SR)
    if kind == 'hp':
        r = 1 / (1 + (freq / np.maximum(fr, 1e-6)) ** (2 * order))
    elif kind == 'lp':
        r = 1 / (1 + (np.maximum(fr, 1e-6) / freq) ** (2 * order))
    elif kind == 'shelf':
        # freq 위쪽을 gain_db 만큼 (부드러운 하이셸프)
        t = 0.5 * (1 + np.tanh((np.log2(np.maximum(fr, 1e-6) / freq)) * 1.2))
        r = 10 ** ((gain_db / 20.0) * t)
    else:
        raise ValueError(kind)
    out = np.empty_like(xp)
    for c in range(xp.shape[1]):
        sp = np.fft.rfft(xp[:, c])
        out[:, c] = np.fft.irfft(sp * r, N)
    return out[pad:pad + n].astype(np.float32)


def fold(x, f=F):
    """꼬리 f초를 머리에 등전력 크로스페이드로 접어 이음매를 없앤다."""
    n = int(L * SR)
    m = int(f * SR)
    raw = x[:n + m]
    out = np.empty((n, raw.shape[1]), dtype=np.float32)
    th = np.linspace(0, np.pi / 2, m, dtype=np.float32)[:, None]
    out[:m] = raw[:m] * np.sin(th) + raw[n:n + m] * np.cos(th)
    out[m:] = raw[m:n]
    out -= out.mean(axis=0, keepdims=True)
    return out


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def lufs(path):
    p = subprocess.run([FF, '-hide_banner', '-nostdin', '-i', path, '-af',
                        'loudnorm=print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True)
    txt = p.stderr
    j = txt[txt.rindex('{'):txt.rindex('}') + 1]
    d = json.loads(j)
    return float(d['input_i']), float(d['input_tp'])


def write_wav(path, x):
    sf.write(path, np.clip(x, -1.5, 1.5), SR, subtype='FLOAT')


def limit(x, ceiling_db=-4.0):
    """아주 부드러운 리미터 — 튀는 순간만 눌러 준다."""
    ceil = 10 ** (ceiling_db / 20.0)
    a = np.abs(x)
    over = a > ceil
    if not over.any():
        return x
    # 8ms 봉투로 부드럽게
    k = int(0.008 * SR)
    env = np.convolve(a.max(axis=1), np.ones(k) / k, 'same')
    g = np.ones_like(env)
    m = env > ceil
    g[m] = ceil / env[m]
    g = np.convolve(g, np.ones(k) / k, 'same')
    return (x * g[:, None]).astype(np.float32)


def ff_filter(src, dst, af):
    subprocess.run([FF, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', src,
                    '-af', af, '-c:a', 'pcm_f32le', dst], check=True)


def build(name, x, fade=F, prefilter=None, lufs_target=TARGET_LUFS):
    loop = fold(x, fade)
    wp = os.path.join(OUT, name + '_pre.wav')
    write_wav(wp, loop)
    if prefilter:
        loop = loop / (np.max(np.abs(loop)) + 1e-9) * 0.6
        write_wav(wp, loop)
        tmp = os.path.join(OUT, name + '_c.wav')
        ff_filter(wp, tmp, prefilter)
        loop = sf.read(tmp, dtype='float32', always_2d=True)[0]
        os.remove(tmp)
        write_wav(wp, loop)          # 압축 뒤 음량을 다시 잰다
    i, tp = lufs(wp)
    g = 10 ** ((lufs_target - i) / 20.0)
    loop = loop * g
    loop = limit(loop)
    mp = os.path.join(OUT, name + '.mp3')
    wp2 = os.path.join(OUT, name + '_post.wav')
    write_wav(wp2, loop)
    subprocess.run([FF, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', wp2,
                    '-c:a', 'libmp3lame', '-b:a', '96k', '-ar', str(SR),
                    '-id3v2_version', '3', '-metadata', 'title=' + name,
                    '-metadata', 'comment=Tody ASMR loop', mp], check=True)
    i2, tp2 = lufs(mp)
    print('%-8s pre %6.1f LUFS  gain %+5.1f dB  → %6.1f LUFS  TP %5.1f  %4.0f KB  rms %.3f'
          % (name, i, 20 * np.log10(g), i2, tp2, os.path.getsize(mp) / 1024, rms(loop)))
    os.remove(wp); os.remove(wp2)
    return mp


# ---------------------------------------------------------------- rain
def make_rain():
    x = load('work_rain_ezwa.wav', 0.2, L + 2.0)
    x = resp_filter(x, 'hp', 70, order=2)
    x = resp_filter(x, 'lp', 9000, order=1)           # 날카로운 숯소리 제거
    x = resp_filter(x, 'shelf', 5500, -3.0)
    return x


# ---------------------------------------------------------------- cafe
def make_cafe():
    bed = load('work_cafe_cc0.wav', 512.0, L + F)
    bed = resp_filter(bed, 'hp', 90, order=2)
    bed = resp_filter(bed, 'lp', 8000, order=1)
    bed = resp_filter(bed, 'shelf', 4000, -4.0)       # 쨍한 고음 눌러 잔잔하게
    lay = load('work_restaurant_stephan.wav', 28.0, L + F)
    lay = resp_filter(lay, 'hp', 250, order=2)
    lay = resp_filter(lay, 'lp', 4800, order=2)       # 식기 소리만 남기고 부드럽게
    lay *= 10 ** (-15.0 / 20.0) * rms(bed) / rms(lay)
    n = min(len(bed), len(lay))
    return bed[:n] + lay[:n]


# ---------------------------------------------------------------- library
def page(src, start, dur, gain_db):
    e = np.zeros((int(L * SR) + int(F * SR), 2), dtype=np.float32)
    s = load(src, start, dur)
    fade = int(0.02 * SR)
    w = np.ones(len(s), dtype=np.float32)
    w[:fade] = np.linspace(0, 1, fade)
    w[-fade:] = np.linspace(1, 0, fade)
    s = s * w[:, None] * 10 ** (gain_db / 20.0)
    return s


def make_library():
    bed = load('work_library_delft.wav', 184.0, L + F)
    bed = resp_filter(bed, 'hp', 55, order=2)
    bed = resp_filter(bed, 'lp', 7000, order=1)
    bed = resp_filter(bed, 'shelf', 3000, 1.5)        # 숨소리처럼 아주 살짝 공기감
    bed = bed / np.max(np.abs(bed)) * 0.5
    out = bed.copy()
    # 책장 넘기는 소리 — 3번, 간격을 고르지 않게. 바닥보다 조금만 위로.
    base = rms(bed)
    turns = [('work_turning_page.wav', 0.55, 1.30, -7.0),
             ('work_old_book.wav', 4.9, 1.05, -8.0),
             ('work_turning_page.wav', 1.95, 0.75, -9.5)]
    spots = [6.4, 19.7, 33.1]
    for (src, st, du, gn), t in zip(turns, spots):
        p = page(src, st, du, 0)
        p /= np.max(np.abs(p)) + 1e-9
        p *= base * 10 ** (gn / 20.0) * 4.0
        i0 = int(t * SR)
        seg = p[:min(len(p), out.shape[0] - i0)]
        out[i0:i0 + len(seg)] += seg
    return out


# ---------------------------------------------------------------- kitchen
def make_kitchen():
    x = load('work_kitchen_cc0.wav', 72.0, L + F)
    x = resp_filter(x, 'hp', 70, order=2)
    x = resp_filter(x, 'lp', 6500, order=2)           # 부딪히는 날카로움 제거
    x = resp_filter(x, 'shelf', 3500, -5.0)
    # 아주 느린 다이내믹 정리: 큰 소리만 살짝 눌러 포근하게
    a = np.abs(x).max(axis=1)
    k = int(0.25 * SR)
    env = np.convolve(a, np.ones(k) / k, 'same')
    ref = np.percentile(env, 70) + 1e-9
    g = np.clip((ref * 1.6) / np.maximum(env, 1e-9), 0.5, 1.0)
    g = np.convolve(g, np.ones(k) / k, 'same')
    return (x * g[:, None]).astype(np.float32)


if __name__ == '__main__':
    build('rain', make_rain(), fade=2.0)
    build('cafe', make_cafe())
    build('library', make_library())
    build('kitchen', make_kitchen(),
          prefilter='acompressor=threshold=0.1:ratio=4:attack=8:release=220:knee=6')
