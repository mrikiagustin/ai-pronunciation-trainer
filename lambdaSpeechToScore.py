
import torch
import json
import os
import WordMatching as wm
import utilsFileIO
import pronunciationTrainer
from phoneme_assessment import get_phoneme_assessor
import base64
import time
import audioread
import numpy as np
from torchaudio.transforms import Resample
import io
import tempfile

trainer_SST_lambda = {}
trainer_SST_lambda['de'] = pronunciationTrainer.getTrainer("de")
trainer_SST_lambda['en'] = pronunciationTrainer.getTrainer("en")

def lambda_handler(event, context):

    data = json.loads(event['body'])
    real_text = data['title']
    encoded_audio = data['base64Audio']
    if ',' in encoded_audio:
        encoded_audio = encoded_audio.split(',', 1)[1]
    file_bytes = base64.b64decode(encoded_audio.encode('utf-8'), validate=True)
    language = data['language']

    if len(real_text) == 0:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp_name = tmp.name

    try:
        tmp.write(file_bytes)
        tmp.flush()

        tmp.close()

        signal, fs = audioread_load(tmp_name)

    except Exception as e:
        print("audioread load error:", e)
        raise

    finally:

        os.remove(tmp_name)

    signal = torch.as_tensor(signal, dtype=torch.float32)
    if signal.ndim == 2:
        # audioread returns [channels, samples] for multi-channel audio.
        signal = signal.mean(dim=0)
    if fs != 16000:
        signal = Resample(orig_freq=fs, new_freq=16000)(signal)
    signal = signal.unsqueeze(0)

    result = trainer_SST_lambda[language].processAudioForGivenText(
        signal, real_text)

    print("result", result)

    start = time.time()
    real_transcripts_ipa = ' '.join(
        [word[0] for word in result['real_and_transcribed_words_ipa']])
    matched_transcripts_ipa = ' '.join(
        [word[1] for word in result['real_and_transcribed_words_ipa']])

    print("start", start)

    real_transcripts = ' '.join(
        [word[0] for word in result['real_and_transcribed_words']])
    matched_transcripts = ' '.join(
        [word[1] for word in result['real_and_transcribed_words']])

    print("real_transcripts", real_transcripts)

    words_real = real_transcripts.lower().split()
    mapped_words = matched_transcripts.split()

    print("mapped_words", mapped_words)

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):

        mapped_letters, mapped_letters_indices = wm.get_best_mapped_words(
            mapped_words[idx], word_real)

        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(
            word_real, mapped_letters)

        is_letter_correct_all_words += ''.join([str(is_correct)
                                                for is_correct in is_letter_correct]) + ' '

    def parse_word_times(value):
        return [
            None if float(item) < 0 else float(item)
            for item in value.split()
        ]

    phoneme_assessment = get_phoneme_assessor().assess(
        signal,
        real_text,
        trainer_SST_lambda[language].ipa_converter,
        fallback_pairs=result['real_and_transcribed_words_ipa'],
        word_start_times=parse_word_times(result['start_time']),
        word_end_times=parse_word_times(result['end_time']),
    )

    phoneme_word_scores = [
        word['score'] for word in phoneme_assessment['words']
    ]
    spoken_ipa = result['recording_ipa']
    if (
        phoneme_assessment['method'] == 'acoustic_ctc_gop'
        and phoneme_assessment['recognized_phones_grouped']
    ):
        spoken_ipa = phoneme_assessment['recognized_phones_grouped']
    pair_accuracy_category = ' '.join(
        [
            str(trainer_SST_lambda[language].getPronunciationCategoryFromAccuracy(score))
            for score in phoneme_word_scores
        ]
    )
    print('Time to post-process results: ', str(time.time()-start))

    res = {'real_transcript': result['recording_transcript'],
           'ipa_transcript': spoken_ipa,
           'asr_ipa_transcript': result['recording_ipa'],
           'pronunciation_accuracy': str(int(phoneme_assessment['overall_score'])),
           'real_transcripts': real_transcripts, 'matched_transcripts': matched_transcripts,
           'real_transcripts_ipa': real_transcripts_ipa, 'matched_transcripts_ipa': matched_transcripts_ipa,
           'pair_accuracy_category': pair_accuracy_category,
           'start_time': result['start_time'],
           'end_time': result['end_time'],
           'is_letter_correct_all_words': is_letter_correct_all_words,
           'phoneme_assessment': phoneme_assessment}

    return json.dumps(res, ensure_ascii=False)




def audioread_load(path, offset=0.0, duration=None, dtype=np.float32):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + \
                (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native

# From Librosa


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
