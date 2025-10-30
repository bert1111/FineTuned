from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio_utils import audio_fft


def read_audio_file(file_name):
    """
    Converts an audio file to an audio signal in numpy array format (used for debugging)

    :param file_name: the name of the audio file to read
    :return: a tuple with the sample rate, the left channel and the right channel audio signal in numpy array format
    """

    # only Wave format supported
    if file_name.endswith('.wav'):
        data, sample_rate = sf.read(file_name, dtype='float32')
        if data.ndim == 1:
            # mono
            left_channel_signal = data
            right_channel_signal = None
        else:
            # stereo
            left_channel_signal = data[:, 0]
            right_channel_signal = data[:, 1]

        return sample_rate, left_channel_signal, right_channel_signal
    else:
        raise Exception('Unsupported Audio Format')


def read_real_time_audio(interval=1):
    """
    Reads real time audio from audio input and yields audio signal in numpy array format

    :return: a tuple with the sample rate and audio signal in numpy array format
    """

    rate = 44100
    chunk_size = int(rate * interval)
    try:
        # Using sounddevice.InputStream for real-time audio input
        with sd.InputStream(samplerate=rate, channels=1, dtype='float32', blocksize=chunk_size) as stream:
            while True:
                data, overflowed = stream.read(chunk_size)
                # data shape is (chunk_size, channels) here channels=1
                np_data = data[:, 0]  # Take the single channel
                yield rate, np_data
    except Exception as e:
        print(f"Error reading audio input: {e}")


def plot_audio_signal(signal, sample_rate, samples=None, plot_max_samples=5000, plot_max_freq=1000):
    """
    Plots an audio signal in audio and frequency domain (used for debugging)

    :param signal: the audio signal in numpy array format
    :param sample_rate: the sample rate of the audio signal
    :param samples: the desired amount of samples in the signal
    :param plot_max_samples: the amount of samples to plot
    :param plot_max_freq: the maximum frequency in the frequency spectrum plot
    """

    if samples is None:
        samples = len(signal)

    fig, axes = plt.subplots(nrows=2, ncols=1, num='Audio Signal Plot')

    # Time Domain Plot
    axes[0].set_title('Time Domain')
    axes[0].set_xlim([0, plot_max_samples])
    axes[0].grid()
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].plot(signal)

    # Frequency Domain Plot

    # Fast Fourier Transform
    frequencies, xf, yf, loudest_frequency, _ = audio_fft(signal, sample_rate, samples)

    axes[1].set_title(
        'Frequency Domain (Fundamental Frequency: ' + str(round(loudest_frequency, 2)) + ' Hz)')
    axes[1].set_xlim([0, plot_max_freq])
    axes[1].grid()
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].plot(xf, 2.0 / samples * np.abs(yf[0:samples // 2]))

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
