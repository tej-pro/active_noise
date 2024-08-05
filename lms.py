import numpy as np
import sys
import wx
from scipy.io import wavfile
import os
import wave
import matplotlib.pyplot as plt
def awgn_noise(x_sig):
    usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower()
    while usr_flag not in ['y', 'n']:
        usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower()
    if usr_flag == 'n':
        return None
    else:
        awgn = np.random.normal(0, 100, size=x_sig.shape)
        x_noised = x_sig + awgn
        return x_noised
def lms(x_sig, d_sig):
    filt_ord = 64
    filt_ord = np.uint16(input("Filter order (default: 64) >> "))
    while filt_ord < 1 and filt_ord%2 != 0:
        print("Value too low or order number is odd.")
        filt_ord = np.uint16(input("Filter order >> "))
    
    # Step size
    mu = 0.0005
    
    # Initialize weights
    w = np.zeros(filt_ord, dtype=np.float32)
    
    # Number of samples of input signal
    sig_len = len(x_sig)
    
    # Initialize estimated filtering
    # est_size = sig_len % filt_ord
    # est = np.zeros(np.uint8(sig_len / filt_ord) + est_size, dtype=np.float32)
    
    # Initialize error rate
    e = np.zeros((sig_len), dtype=np.float64)
    
    
    # LMS algorithm
    for i in range(filt_ord, sig_len):
        # BUG: 1st array element of input singal wil not be processed.
        # Get from last to first value of signal's window
        window_sig = x_sig[i:i-filt_ord:-1]
        
        # Filter with previous coefficients
        est = w * window_sig
        
        # Evaluate error rate
        e[i:i-filt_ord:-1] = d_sig[i:i-filt_ord:-1] - est
        
        # Estimate next coefficients
        w = w * 2 * mu * window_sig * e[i-filt_ord]
        
    # Filter input signal with adaptive coefficients
    #filt_out = adapt_filt(w, x_sig, filt_ord, sig_len)
    filt_out = np.convolve(x_sig, w)
    
    # Ask user for learning rate visualization
    plot_flg = 'y'
    plot_flg = input("Visualize learning rate? [Y/n] >> ").lower()
    
    # Sanity check for user input
    if plot_flg != 'y' and plot_flg != 'n':
        plot_flg = input("Visualize learning rate? [Y/n] >> ").lower()
    
    v = [filt_out,e]
    return v
sample_rate, signal = wavfile.read('/home/rguktrkvalley/Downloads/file.wav')
read_path = os.path.abspath(os.path.join('/home/rguktrkvalley/Downloads/file.wav', os.pardir))
signal = signal[:4410,0]
noise_signal = awgn_noise(signal)
filtered_data = lms(noise_signal,signal)
plt.figure(1)
plt.title("INPUT SIGNAL")
plt.plot(signal)
plt.savefig('/home/rguktrkvalley/Downloads/input_signal.png')
plt.figure(2)
plt.title("NOISE SIGNAL")
plt.plot(noise_signal)
plt.savefig('/home/rguktrkvalley/Downloads/noise_signal.png')
plt.figure(3)
plt.title("ESTIMATED SIGNAL")
plt.plot(filtered_data[1])
plt.savefig('/home/rguktrkvalley/Downloads/output_signal.png')
plt.show()    
wavfile.write('/home/rguktrkvalley/Downloads/output.wav', sample_rate,filtered_data[1])

    
