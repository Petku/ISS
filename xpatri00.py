import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from scipy.signal import lfilter, freqz, tf2zpk
from scipy.io import wavfile

# Nacitanie signalu a jeho normalizacia
fs, data = wavfile.read('xpatri00.wav')
data = data / 2**15

##################################################
# Excercise 1/2 ##################################
##################################################
print('Exercise 1/2')

# Vypis vzorkovacej frekvencie a poctu vzorkov
print(f'Vzorkovacia frekvencia: {fs} Hz\nPocet vzorkov:{data.size}\nPocet bin.symbolov:{data.size//16}')
plot_data = data[:320]

# Vypocet casu
t = np.arange(plot_data.size) / fs
# plt.figure(figsize=(6, 4))

discrethe_data = list()

for i in range(7, data.size, 16):
    if data[i] > 0:
        discrethe_data.append(1)
    else:
        discrethe_data.append(0)

fh = open('xpatri00.txt', 'r')

flag_xor = True
for lino, line in enumerate(fh):
    if(discrethe_data[lino] != int(line)):
        flag_xor = False
        print('Vygenerovane binarne symboli sa nezhoduju so symbolmi v subore xpatri00.txt')
        break

if flag_xor:
    print('Vygenerovane binarne symboli sa zhoduju so symbolmi v subore xpatri00.txt')

fh.close()

t2 = np.arange(0.5, len(discrethe_data)//100) / 1000

plt.plot(t, plot_data, linewidth=0.8)
plt.stem(t2, discrethe_data[:len(discrethe_data)//100], 'k--', basefmt=' ')
plt.grid()


plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$s[n]$')
plt.gca().set_title('Excercise 1/2')

plt.tight_layout()
plt.show()
plt.close()

##################################################
# Excercise 3 ####################################
##################################################
print('Exercise 3')

def zplane(z, p, filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed', alpha=0.5)
    ax.add_patch(uc)
    axvline(0, color='0.7')
    axhline(0, color='0.7')

    # Plot the poles and set marker properties
    plt.plot(p.real, p.imag, 'rx', markersize=9, alpha=0.5, label='póly')

    # Plot the zeros and set marker properties
    plt.plot(z.real, z.imag, 'o', markersize=9,
                     color='none', alpha=0.5,
                     markeredgecolor='b', label='nuly'
                     )


    # set the ticks
    r = 1.5
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, 0, .5, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)

    if filename is None:
        plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
        plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')
        plt.gca().set_title('Excercise 3')
        plt.grid(linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
    else:
        plt.savefig(filename)

    return z, p

b = [0.0192, -0.0185, -0.0185, 0.0192]
a = [1.0000, -2.8870, 2.7997, -0.9113]

zeros, poles, _ = tf2zpk(b, a)
zplane(zeros, poles)

is_stable = (poles.size == 0) or np.all(np.abs(poles) < 1)
print('Filtr {} stabilní.'.format('je' if is_stable else 'není'))

##################################################
# Excercise 4 ####################################
##################################################
print('Exercise 4')

# frekvencni charakteristika
w, H = freqz(b, a)

# f0 = 483

f0 = np.min(np.abs(H[:100]))
f0_index = 0

for i, h in enumerate(np.abs(H)):
    if f0 == h:
        f0_index = i
        break

f0 = w[f0_index] / 2 / np.pi * fs



print(f'Mezni frekvence je f0 = {f0}')
axvline(f0, color='orange',linestyle='--', linewidth=0.8, label='Mezni frekvence')

plt.plot(w / 2 / np.pi * fs, np.abs(H), linewidth=0.8)
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_title('Excercise 4\nModul frekvenční charakteristiky $|H(e^{j\omega})|$')
plt.grid(linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.show()
plt.close()

##################################################
# Excercise 5 ####################################
##################################################

filtered_data = lfilter(b, a, data)
plot_filtered_data = filtered_data[:320]

axhline(0, color='0.7')
plt.plot(t, plot_data, 'c', linewidth=0.8, label='$s[n]$')
plt.plot(t, plot_filtered_data, linewidth=0.8, label='$ss[n]$')
plt.grid(linestyle='--', alpha=0.5)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$s[n], ss[n], ss_{shifted}[n]$')
plt.gca().set_title('Excercise 5/6')


##################################################
# Excercise 6 ####################################
##################################################

filtered_data_shifted = list()
shift = 16
for i in range(shift, filtered_data.size):
    filtered_data_shifted.append(filtered_data[i])

plot_filtered_data_shifted = filtered_data_shifted[:320]
plt.plot(t, plot_filtered_data_shifted, 'k', linewidth=0.8, alpha=0.5, label='$ss_{shifted}[n]$')

binary_shifted_data = list()
for i in range(7, len(filtered_data_shifted), 16):
    if filtered_data_shifted[i] > 0:
        binary_shifted_data.append(1)
    else:
        binary_shifted_data.append(0)

plot_binary_shifted_data = binary_shifted_data[:20]

markerline, stemlines, baseline = plt.stem(t2, plot_binary_shifted_data, label="$ss_{shifted}[n]_{binary}$")

plt.setp(stemlines, color='orange', linestyle=':', alpha=0.4)
plt.setp(markerline, color='none', alpha=0.8, markeredgecolor='orange')
plt.setp(baseline, alpha=0)

plt.legend(loc='lower center', ncol=4)
plt.show()
plt.close()

##################################################
# Excercise 7 ####################################
##################################################
print('Excercise 7')

bad_binary_symbol_count = 0
fh = open('xpatri00.txt','r')
for lino,line in enumerate(fh):
    if lino >= len(binary_shifted_data):
        break

    if(binary_shifted_data[lino] != int(line)):
        bad_binary_symbol_count +=1

rate_of_binarysymbol_errors = bad_binary_symbol_count / len(binary_shifted_data) * 100

print(f'Z celkoveho poctu vzrokov {len(binary_shifted_data)} bolo chybnych {bad_binary_symbol_count}')
print(f'Celkova chybovost je teda: {rate_of_binarysymbol_errors:.5} %')

fh.close()


##################################################
# Excercise 8 ####################################
##################################################

spec_data = np.abs(np.fft.fft(data))
spec_filtered_data = np.abs(np.fft.fft(filtered_data))

fshalf = np.arange(fs/2)

plt.plot(fshalf, spec_data[:fs//2], label='Modul spektra signalu $s[n]$')
plt.plot(fshalf, spec_filtered_data[:fs//2], label='Modul spektra signalu $ss[n]$')
plt.gca().set_xlabel('$f[Hz]$')
plt.gca().set_title(f'Exercise 8\nModuly spektier signalov $s[n]$ a $ss[n]$')

plt.gca().grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')
plt.show()
plt.close()


##################################################
# Excercise 9 ####################################
##################################################
print('Excercise 9')

xmin = np.min(data)
xmax = np.max(data)
n_aprx = 50
x = np.linspace(xmin, xmax, n_aprx)
binsize = np.abs(xmax - xmin)
hist, _ = np.histogram(data, n_aprx)
px = hist / data.size / binsize

plt.figure(figsize=(8,3))
plt.plot(x, px)
plt.gca().set_xlabel('$x$')
plt.gca().set_title(f'Exercise 9\nOdhad funkce hustoty rozdělení pravděpodobnosti $p(x)$')

plt.gca().grid(alpha=0.5, linestyle='--')
plt.show()
plt.close()

print(f'Integral odhadu pravdivostnej funckie:{np.sum(px*binsize)}')

##################################################
# Excercise 10 ###################################
##################################################

Rv = np.correlate(data, data, 'full') / data.size

plt.plot(np.arange(-50, 50), Rv[Rv.size//2-50:Rv.size//2+50])
plt.xticks([-50, -40, -30, -20, -10, 0, 10, 20, 30 , 40, 50])
plt.gca().set_title('Excercise 10\nVychyleny odhad autokorelacnich koeficientu')
plt.gca().set_xlabel('$k$')
plt.gca().set_ylabel('$R[k]$')

plt.gca().grid(alpha=0.5, linestyle='--')
plt.show()
plt.close()

##################################################
# Excercise 11 ###################################
##################################################
print('Excercise 11')

print(f'Hodnota R[0] = {Rv[Rv.size//2]}')
print(f'R[1] = {Rv[Rv.size//2+1]}')
print(f'R[16] = {Rv[Rv.size//2+16]}')

##################################################
# Excercise 12 ###################################
##################################################

px1x2, x1_edges, x2_edges = np.histogram2d(data[:data.size-1], data[1:], n_aprx, normed=True)

X, Y = np.meshgrid(x1_edges, x2_edges)
im = plt.pcolormesh(X, Y, px1x2)

cbar = plt.colorbar(im)
cbar.set_label('$p(x_1,x_2,1)$', rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=8)
plt.gca().tick_params(labelsize=8)

plt.gca().set_xlabel('$x_1$')
plt.gca().set_ylabel('$x_2$')
plt.gca().set_title('Excercise 12\nSdruzena funkce hustoty rozdeleni pravdepodobnosti mezi casy $n$, $n+1$')
plt.show()


##################################################
# Excercise 13 ###################################
##################################################
print('Excercise 13')


binsize = np.abs(x1_edges[0] - x1_edges[1]) * np.abs(x2_edges[0] - x2_edges[1])
integral = np.sum(px1x2 * binsize)
print(f'Vysledok integralu px1x2 funckie je {integral}')

##################################################
# Excercise 14 ###################################
##################################################
print('Excercise 14')

bin_centers_x1 = x1_edges[:-1] + (x1_edges[1:] - x1_edges[:-1]) / 2
bin_centers_x2 = x2_edges[:-1] + (x2_edges[1:] - x2_edges[:-1]) / 2
x1x2 = np.outer(bin_centers_x1, bin_centers_x2)
R = np.sum(x1x2 * px1x2 * binsize)

print(f'Korelacni koeficient sa rovna R[1] = {R}')
